"""
Window Capture — Per-window enumeration and frame capture for tab-detach streaming.

Instead of full-screen capture, this module captures individual OS windows so each
remote application (Notepad, CMD, etc.) can be streamed as a separate session.

Backends (cross-platform, guarded imports):
  Windows: win32gui EnumWindows + GetWindowDC/BitBlt, fallback mss region
  Linux: Xlib _NET_CLIENT_LIST or xdotool, fallback mss region

Follows FrameCapture contract: capture_frame() → JPEG bytes, capture_loop() → generator.

Reuses:
  - frame_capture.py: FrameConfig, _CaptureCircuitBreaker, _encode_pil_image pattern
  - frame_capture.py:54-63: compute_frame_difference() for skip-unchanged
"""

import io
import logging
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

logger = logging.getLogger('hevolve.remote_desktop')

# ── Optional dependencies (guarded imports) ─────────────────────

_mss = None
_PIL_Image = None

try:
    import mss as _mss_module
    _mss = _mss_module
except ImportError:
    pass

try:
    from PIL import Image as _PIL_Image_module
    _PIL_Image = _PIL_Image_module
except ImportError:
    pass

# Windows-specific (win32gui, win32ui, win32con, win32api, win32process)
_win32gui = None
_win32ui = None
_win32con = None
_win32api = None
_win32process = None

try:
    import win32gui as _win32gui_mod
    import win32ui as _win32ui_mod
    import win32con as _win32con_mod
    import win32api as _win32api_mod
    import win32process as _win32process_mod
    _win32gui = _win32gui_mod
    _win32ui = _win32ui_mod
    _win32con = _win32con_mod
    _win32api = _win32api_mod
    _win32process = _win32process_mod
except ImportError:
    pass

# Linux-specific (Xlib)
_Xlib_display = None
try:
    from Xlib import display as _Xlib_display_mod
    _Xlib_display = _Xlib_display_mod
except ImportError:
    pass


# ── Data Structures ────────────────────────────────────────────

@dataclass
class WindowInfo:
    """Metadata for a single OS window.

    The trailing ``z_order`` / ``is_foreground`` / ``is_occluded`` /
    ``occluded_pct`` / ``is_protected`` / ``monitor_idx`` fields were
    added in Phase 1 of the VLM best-of-all-worlds plan (memory/
    vlm_best_of_all_worlds_plan.md §1).  All have safe defaults so
    existing callers (window_session, dlna_bridge, agent_tools) keep
    working without modification.
    """
    hwnd: int                           # Window handle (HWND on Windows, XID on Linux)
    title: str
    process_name: str
    pid: int
    rect: Tuple[int, int, int, int]     # (x, y, width, height)
    visible: bool = True
    minimized: bool = False
    # ── Phase-1 additions (VLM occlusion + multi-monitor) ──
    z_order: int = 0                    # 0 = topmost; higher = further back
    is_foreground: bool = False         # True if this is the active window
    is_occluded: bool = False           # True if any other window covers > 5% of rect
    occluded_pct: float = 0.0           # 0.0–100.0; % of rect area covered
    is_protected: bool = False          # DWM-cloaked (DRM, virtual desktop hidden)
    monitor_idx: int = -1               # Index into list_monitors() (-1 = unknown)

    def to_dict(self) -> dict:
        return {
            'hwnd': self.hwnd,
            'title': self.title,
            'process_name': self.process_name,
            'pid': self.pid,
            'rect': list(self.rect),
            'visible': self.visible,
            'minimized': self.minimized,
            'z_order': self.z_order,
            'is_foreground': self.is_foreground,
            'is_occluded': self.is_occluded,
            'occluded_pct': round(self.occluded_pct, 1),
            'is_protected': self.is_protected,
            'monitor_idx': self.monitor_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WindowInfo':
        return cls(
            hwnd=d['hwnd'],
            title=d['title'],
            process_name=d.get('process_name', ''),
            pid=d.get('pid', 0),
            rect=tuple(d.get('rect', (0, 0, 0, 0))),
            visible=d.get('visible', True),
            minimized=d.get('minimized', False),
            z_order=d.get('z_order', 0),
            is_foreground=d.get('is_foreground', False),
            is_occluded=d.get('is_occluded', False),
            occluded_pct=d.get('occluded_pct', 0.0),
            is_protected=d.get('is_protected', False),
            monitor_idx=d.get('monitor_idx', -1),
        )


@dataclass
class WindowCaptureConfig:
    """Configuration for per-window capture."""
    quality: int = 80              # JPEG quality (1-100)
    scale_factor: float = 1.0     # Downscale factor
    max_fps: int = 30
    min_change_threshold: float = 0.01
    keyframe_interval: int = 30
    adaptive_interval: bool = True
    max_backoff_seconds: float = 2.0


# ── Window Enumerator ──────────────────────────────────────────

class WindowEnumerator:
    """Cross-platform window enumeration.

    Windows: win32gui.EnumWindows + win32gui.GetWindowText
    Linux: Xlib _NET_CLIENT_LIST or xdotool fallback
    """

    def __init__(self):
        self._system = platform.system()

    def list_windows(self, include_minimized: bool = False) -> List[WindowInfo]:
        """List all visible application windows on the host.

        Args:
            include_minimized: Include minimized/iconic windows.

        Returns:
            List of WindowInfo for each visible window.
        """
        if self._system == 'Windows' and _win32gui:
            return self._list_windows_win32(include_minimized)
        elif self._system == 'Linux':
            return self._list_windows_linux(include_minimized)
        return []

    def get_window_by_title(self, title_pattern: str) -> Optional[WindowInfo]:
        """Find window by title substring or regex pattern."""
        windows = self.list_windows(include_minimized=True)
        pattern = re.compile(title_pattern, re.IGNORECASE)
        for w in windows:
            if pattern.search(w.title):
                return w
        return None

    def get_window_by_pid(self, pid: int) -> Optional[WindowInfo]:
        """Find the primary window for a process ID."""
        windows = self.list_windows(include_minimized=True)
        for w in windows:
            if w.pid == pid:
                return w
        return None

    def refresh_window_info(self, window: WindowInfo) -> Optional[WindowInfo]:
        """Refresh a window's position/visibility (handle may have moved)."""
        if self._system == 'Windows' and _win32gui:
            return self._refresh_win32(window)
        elif self._system == 'Linux':
            return self._refresh_linux(window)
        return None

    # ── Windows backend ────────────────────────────────────────

    def _list_windows_win32(self, include_minimized: bool) -> List[WindowInfo]:
        """Enumerate windows via Win32 API.

        EnumWindows yields windows in **top-to-bottom z-order** — the first
        callback invocation is the topmost window.  We use that order to
        populate ``z_order`` (0 = topmost) and to compute ``is_occluded`` /
        ``occluded_pct`` in :func:`_compute_occlusion`.
        """
        results = []
        try:
            foreground_hwnd = _win32gui.GetForegroundWindow()
        except Exception:
            foreground_hwnd = 0

        def enum_callback(hwnd, _):
            if not _win32gui.IsWindowVisible(hwnd):
                return
            title = _win32gui.GetWindowText(hwnd)
            if not title:
                return

            minimized = bool(_win32gui.IsIconic(hwnd))
            if minimized and not include_minimized:
                return

            # Get window rect
            try:
                left, top, right, bottom = _win32gui.GetWindowRect(hwnd)
                width = right - left
                height = bottom - top
                if width <= 0 or height <= 0:
                    return
            except Exception:
                return

            # Get process info
            pid = 0
            process_name = ''
            try:
                _, pid = _win32process.GetWindowThreadProcessId(hwnd)
                process_name = self._get_process_name_win32(pid)
            except Exception:
                pass

            # Phase-1 enrichment.  z_order is just the EnumWindows arrival
            # index (top = 0).  is_protected uses DWMWA_CLOAKED — true for
            # DRM-protected windows (Netflix, banking apps that opt out)
            # AND for virtual-desktop-hidden windows (cloaked while not on
            # current desktop).  Either way, capture_window will return
            # black pixels, so the flag warns callers to fall back.
            results.append(WindowInfo(
                hwnd=hwnd,
                title=title,
                process_name=process_name,
                pid=pid,
                rect=(left, top, width, height),
                visible=True,
                minimized=minimized,
                z_order=len(results),
                is_foreground=(hwnd == foreground_hwnd),
                is_protected=_is_dwm_cloaked(hwnd),
            ))

        _win32gui.EnumWindows(enum_callback, None)
        # Compute occlusion + monitor assignment in a second pass — both
        # need the full window list / monitor list to make sense.
        _compute_occlusion(results)
        try:
            _assign_monitors(results, list_monitors())
        except Exception as e:
            logger.debug(f"Monitor assignment skipped: {e}")
        return results

    def _get_process_name_win32(self, pid: int) -> str:
        """Get process name from PID on Windows."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                          False, pid)
            if handle:
                try:
                    import os
                    import ctypes.wintypes
                    buf = ctypes.create_unicode_buffer(260)
                    size = ctypes.wintypes.DWORD(260)
                    kernel32.QueryFullProcessImageNameW(handle, 0,
                                                       buf, ctypes.byref(size))
                    full_path = buf.value
                    return os.path.basename(full_path) if full_path else ''
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:
            pass
        return ''

    def _refresh_win32(self, window: WindowInfo) -> Optional[WindowInfo]:
        """Refresh window info for an existing handle."""
        hwnd = window.hwnd
        try:
            if not _win32gui.IsWindow(hwnd):
                return None
            title = _win32gui.GetWindowText(hwnd)
            visible = bool(_win32gui.IsWindowVisible(hwnd))
            minimized = bool(_win32gui.IsIconic(hwnd))
            left, top, right, bottom = _win32gui.GetWindowRect(hwnd)
            return WindowInfo(
                hwnd=hwnd,
                title=title,
                process_name=window.process_name,
                pid=window.pid,
                rect=(left, top, right - left, bottom - top),
                visible=visible,
                minimized=minimized,
            )
        except Exception:
            return None

    # ── Linux backend ──────────────────────────────────────────

    def _list_windows_linux(self, include_minimized: bool) -> List[WindowInfo]:
        """Enumerate windows via xdotool (most portable) or Xlib."""
        # Try xdotool first (works on both X11 and some Wayland setups)
        try:
            return self._list_windows_xdotool(include_minimized)
        except Exception:
            pass

        # Fallback to Xlib
        if _Xlib_display:
            try:
                return self._list_windows_xlib(include_minimized)
            except Exception:
                pass

        return []

    def _list_windows_xdotool(self, include_minimized: bool) -> List[WindowInfo]:
        """Enumerate via xdotool search + getwindowgeometry."""
        output = subprocess.check_output(
            ['xdotool', 'search', '--name', '.'],
            timeout=5,
            text=True,
        )
        results = []
        for line in output.strip().split('\n'):
            xid_str = line.strip()
            if not xid_str:
                continue
            try:
                xid = int(xid_str)
            except ValueError:
                continue

            info = self._get_xdotool_window_info(xid, include_minimized)
            if info:
                results.append(info)
        return results

    def _get_xdotool_window_info(self, xid: int,
                                  include_minimized: bool) -> Optional[WindowInfo]:
        """Get window info for a single XID via xdotool."""
        try:
            name_out = subprocess.check_output(
                ['xdotool', 'getwindowname', str(xid)],
                timeout=2, text=True,
            ).strip()
        except Exception:
            return None

        if not name_out:
            return None

        try:
            geo_out = subprocess.check_output(
                ['xdotool', 'getwindowgeometry', '--shell', str(xid)],
                timeout=2, text=True,
            )
        except Exception:
            return None

        # Parse geometry: X=, Y=, WIDTH=, HEIGHT=
        geo = {}
        for gline in geo_out.strip().split('\n'):
            if '=' in gline:
                k, v = gline.split('=', 1)
                geo[k.strip()] = int(v.strip())

        x = geo.get('X', 0)
        y = geo.get('Y', 0)
        w = geo.get('WIDTH', 0)
        h = geo.get('HEIGHT', 0)
        if w <= 0 or h <= 0:
            return None

        # Get PID
        pid = 0
        try:
            pid_out = subprocess.check_output(
                ['xdotool', 'getwindowpid', str(xid)],
                timeout=2, text=True,
            ).strip()
            pid = int(pid_out)
        except Exception:
            pass

        # Get process name from PID
        process_name = ''
        if pid:
            try:
                cmd_out = subprocess.check_output(
                    ['ps', '-p', str(pid), '-o', 'comm='],
                    timeout=2, text=True,
                ).strip()
                process_name = cmd_out
            except Exception:
                pass

        return WindowInfo(
            hwnd=xid,
            title=name_out,
            process_name=process_name,
            pid=pid,
            rect=(x, y, w, h),
            visible=True,
            minimized=False,
        )

    def _list_windows_xlib(self, include_minimized: bool) -> List[WindowInfo]:
        """Enumerate via python-xlib _NET_CLIENT_LIST."""
        disp = _Xlib_display.Display()
        root = disp.screen().root

        # Get _NET_CLIENT_LIST atom
        client_list_atom = disp.intern_atom('_NET_CLIENT_LIST')
        prop = root.get_full_property(client_list_atom, 0)
        if not prop:
            disp.close()
            return []

        results = []
        for xid in prop.value:
            try:
                win = disp.create_resource_object('window', xid)
                name = win.get_wm_name() or ''
                if not name:
                    continue
                geo = win.get_geometry()
                pid = 0
                pid_atom = disp.intern_atom('_NET_WM_PID')
                pid_prop = win.get_full_property(pid_atom, 0)
                if pid_prop:
                    pid = pid_prop.value[0]

                results.append(WindowInfo(
                    hwnd=xid,
                    title=name,
                    process_name='',
                    pid=pid,
                    rect=(geo.x, geo.y, geo.width, geo.height),
                    visible=True,
                    minimized=False,
                ))
            except Exception:
                continue

        disp.close()
        return results

    def _refresh_linux(self, window: WindowInfo) -> Optional[WindowInfo]:
        """Refresh window info on Linux."""
        return self._get_xdotool_window_info(window.hwnd, True)


# ── Per-Window Frame Capture ──────────────────────────────────

class WindowCapture:
    """Capture a specific window (not full screen).

    Follows FrameCapture contract: capture_frame() → JPEG bytes.
    Uses mss region capture with window rect as the capture area.

    Windows: Prefers win32gui GetWindowDC + BitBlt (captures even occluded windows).
    Linux/fallback: mss region capture (only works if window is visible).
    """

    def __init__(self, window_info: WindowInfo,
                 config: Optional[WindowCaptureConfig] = None):
        self._window = window_info
        self.config = config or WindowCaptureConfig()
        self._running = False
        self._last_frame: Optional[bytes] = None
        self._frame_count = 0
        self._mss_instance = None
        self._system = platform.system()

    @property
    def window_info(self) -> WindowInfo:
        return self._window

    def capture_frame(self) -> Optional[bytes]:
        """Capture single frame of this window as JPEG bytes."""
        # Try win32 (can capture occluded windows)
        if self._system == 'Windows' and _win32gui:
            try:
                frame = self._capture_win32()
                if frame:
                    return frame
            except Exception as e:
                logger.debug(f"Win32 window capture failed: {e}")

        # Fallback: mss region capture (window must be visible)
        if _mss:
            try:
                frame = self._capture_mss_region()
                if frame:
                    return frame
            except Exception as e:
                logger.debug(f"MSS region capture failed: {e}")

        return None

    def capture_loop(self) -> Generator[bytes, None, None]:
        """Yield JPEG frames of this window (same contract as FrameCapture)."""
        self._running = True
        interval = 1.0 / self.config.max_fps
        adaptive_interval = interval
        self._frame_count = 0

        try:
            from integrations.vision.frame_store import compute_frame_difference
        except ImportError:
            def compute_frame_difference(f1, f2):
                if len(f1) != len(f2):
                    return 1.0
                diff = sum(abs(a - b) for a, b in zip(f1[:1000], f2[:1000]))
                return min(diff / (255 * min(len(f1), 1000)), 1.0)

        try:
            while self._running:
                start = time.monotonic()

                frame = self.capture_frame()
                if frame is None:
                    time.sleep(interval)
                    continue

                self._frame_count += 1

                # Skip unchanged frames (unless keyframe)
                is_keyframe = (self._frame_count %
                               self.config.keyframe_interval == 0)
                if self._last_frame and not is_keyframe:
                    try:
                        diff = compute_frame_difference(
                            self._last_frame[:4096], frame[:4096])
                        if diff < self.config.min_change_threshold:
                            if self.config.adaptive_interval:
                                adaptive_interval = min(
                                    adaptive_interval * 1.5,
                                    self.config.max_backoff_seconds,
                                )
                            elapsed = time.monotonic() - start
                            time.sleep(max(0, adaptive_interval - elapsed))
                            continue
                    except Exception:
                        pass

                adaptive_interval = interval
                self._last_frame = frame
                yield frame

                elapsed = time.monotonic() - start
                time.sleep(max(0, interval - elapsed))
        finally:
            self._running = False
            self._cleanup()

    def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_window_info(self) -> WindowInfo:
        """Return current window metadata (position may have changed)."""
        enum = WindowEnumerator()
        refreshed = enum.refresh_window_info(self._window)
        if refreshed:
            self._window = refreshed
        return self._window

    def get_stats(self) -> dict:
        return {
            'running': self._running,
            'frame_count': self._frame_count,
            'window': self._window.to_dict(),
            'config': {
                'max_fps': self.config.max_fps,
                'quality': self.config.quality,
                'scale_factor': self.config.scale_factor,
            },
        }

    # ── Windows capture backend ────────────────────────────────

    def _capture_win32(self) -> Optional[bytes]:
        """Capture window via Win32 GDI (works even if window is behind others)."""
        hwnd = self._window.hwnd
        if not _win32gui.IsWindow(hwnd):
            return None

        # Get client area dimensions
        left, top, right, bottom = _win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None

        # Create device contexts
        hwnd_dc = _win32gui.GetWindowDC(hwnd)
        mfc_dc = _win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create bitmap
        bitmap = _win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        # BitBlt: copy window content to bitmap
        # PW_RENDERFULLCONTENT = 0x00000002 for layered windows
        try:
            result = save_dc.BitBlt(
                (0, 0), (width, height), mfc_dc,
                (left, top), _win32con.SRCCOPY,
            )
        except Exception:
            result = False

        if not result and result is not None:
            # Cleanup on failure
            _win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            _win32gui.ReleaseDC(hwnd, hwnd_dc)
            return None

        # Extract bitmap data
        bmp_info = bitmap.GetInfo()
        bmp_data = bitmap.GetBitmapBits(True)

        # Cleanup GDI objects
        _win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        _win32gui.ReleaseDC(hwnd, hwnd_dc)

        # Convert to JPEG via PIL
        if _PIL_Image and bmp_data:
            try:
                img = _PIL_Image.frombuffer(
                    'RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']),
                    bmp_data, 'raw', 'BGRX', 0, 1,
                )
                return self._encode_pil_image(img)
            except Exception as e:
                logger.debug(f"PIL conversion failed: {e}")

        return None

    # ── MSS region capture backend ─────────────────────────────

    def _capture_mss_region(self) -> Optional[bytes]:
        """Capture window region via mss (cross-platform, window must be visible)."""
        if self._mss_instance is None:
            self._mss_instance = _mss.mss()

        x, y, w, h = self._window.rect
        if w <= 0 or h <= 0:
            return None

        monitor = {'left': x, 'top': y, 'width': w, 'height': h}
        sct_img = self._mss_instance.grab(monitor)

        if _PIL_Image:
            img = _PIL_Image.frombytes('RGB', sct_img.size,
                                       sct_img.bgra, 'raw', 'BGRX')
            return self._encode_pil_image(img)
        return _mss.tools.to_png(sct_img.rgb, sct_img.size)

    # ── Encoding (matches FrameCapture._encode_pil_image) ──────

    def _encode_pil_image(self, img) -> bytes:
        """Encode PIL Image to JPEG bytes with configured quality and scale."""
        if self.config.scale_factor != 1.0:
            new_size = (
                int(img.width * self.config.scale_factor),
                int(img.height * self.config.scale_factor),
            )
            img = img.resize(new_size,
                             _PIL_Image.LANCZOS if _PIL_Image else 1)

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.config.quality, optimize=True)
        return buf.getvalue()

    def _cleanup(self) -> None:
        """Release capture resources."""
        if self._mss_instance:
            try:
                self._mss_instance.close()
            except Exception:
                pass
            self._mss_instance = None


# ════════════════════════════════════════════════════════════════════
# Phase 1 of vlm_best_of_all_worlds_plan.md §1 — module-level helpers
# the VLM stack uses for occlusion-tolerant capture and multi-monitor
# enumeration.  Lives in this file (not a sibling) so we have ONE
# canonical home for window enumeration; the VLM stack imports from
# here rather than maintaining a parallel implementation (Gate 4).
# ════════════════════════════════════════════════════════════════════

DWMWA_CLOAKED = 14  # DWM window-attribute index — non-zero = cloaked


def _is_dwm_cloaked(hwnd) -> bool:
    """True if the window is DWM-cloaked.

    Cloaked windows include:
      * DRM-protected content (Netflix desktop app, some banking apps
        opted out of capture) — PrintWindow/BitBlt return black pixels
      * Windows on other virtual desktops (cloaked while not on current
        desktop) — capturing them returns last-frame snapshot, often stale

    Either way, callers should be told the capture won't reflect live
    content and they may want a different fallback.  Best-effort: if
    dwmapi isn't available (very old Windows), return False.
    """
    if not _win32gui:
        return False
    try:
        import ctypes
        cloaked = ctypes.c_int(0)
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            int(hwnd), DWMWA_CLOAKED,
            ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return result == 0 and cloaked.value != 0
    except Exception:
        return False


def _compute_occlusion(windows: List[WindowInfo]) -> None:
    """Annotate each window's ``is_occluded`` / ``occluded_pct`` in place.

    Assumes windows are sorted top-to-bottom z-order (the EnumWindows
    callback order).  For each window, compute the union of intersections
    with every window above it; cap at the window's own area to avoid
    over-counting when multiple windows above overlap each other AND this
    window.  Threshold for is_occluded = > 5% covered (lets small overlay
    bars / tray windows not count as 'occluded').
    """
    for i, win in enumerate(windows):
        if win.minimized:
            continue
        wx, wy, ww, wh = win.rect
        if ww <= 0 or wh <= 0:
            continue
        win_area = ww * wh
        overlap_area = 0
        for j in range(i):
            other = windows[j]
            if other.minimized:
                continue
            ox, oy, ow, oh = other.rect
            ix1 = max(wx, ox)
            iy1 = max(wy, oy)
            ix2 = min(wx + ww, ox + ow)
            iy2 = min(wy + wh, oy + oh)
            if ix1 < ix2 and iy1 < iy2:
                overlap_area += (ix2 - ix1) * (iy2 - iy1)
        # Cap so a window covered by 2 things doesn't read as 200%.
        overlap_area = min(overlap_area, win_area)
        win.occluded_pct = (overlap_area / win_area) * 100.0
        win.is_occluded = win.occluded_pct > 5.0


def _assign_monitors(windows: List[WindowInfo],
                     monitors: List[dict]) -> None:
    """Set each window's ``monitor_idx`` based on which monitor its
    rect's center point falls on.  Monitors that fully contain the
    window's center beat partial-overlap monitors (avoids ambiguity
    for windows straddling two monitors)."""
    for win in windows:
        wx, wy, ww, wh = win.rect
        cx, cy = wx + ww // 2, wy + wh // 2
        win.monitor_idx = -1
        for m in monitors:
            mx, my, mw, mh = m['rect']
            if mx <= cx < mx + mw and my <= cy < my + mh:
                win.monitor_idx = m['idx']
                break


def _ensure_dpi_aware_for_enum() -> None:
    """Make this process DPI-aware before reading monitor / window rects.

    Without this, EnumDisplayMonitors / GetWindowRect return LOGICAL
    coordinates (post-scaling), e.g. 1707×960 instead of the physical
    2560×1440 on a 150%-scaled display.  Pyautogui's screenshot returns
    PHYSICAL pixels, so any caller mixing the two would have its click
    coordinates land in the wrong spot — exactly the bug
    local_computer_tool's _ensure_dpi_aware() was added to prevent.

    Safe / idempotent: SetProcessDpiAwareness is per-process; calling
    twice with the same value is a no-op, calling with a different
    value silently fails (so existing DPI-aware processes aren't
    disturbed).
    """
    if platform.system() != 'Windows':
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2 (Win 8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def list_monitors() -> List[dict]:
    """Enumerate physical displays.

    Returns:
        List of dicts: ``[{idx, rect: (x,y,w,h), scale_factor,
        is_primary, name}]``.  ``rect`` is in **physical** pixel
        coords thanks to :func:`_ensure_dpi_aware_for_enum`.
        Negative values are valid for monitors left/above the primary.
        ``scale_factor`` is the DPI scale (1.0 = 96 DPI; 1.5 = 144 DPI
        / 150% scaling).  Empty list on non-Windows or unsupported
        builds (Phase 2 adds macOS/Linux).
    """
    if platform.system() != 'Windows':
        return []
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return []

    _ensure_dpi_aware_for_enum()

    MONITORINFOF_PRIMARY = 0x00000001

    class MONITORINFOEX(ctypes.Structure):
        _fields_ = [
            ('cbSize', wintypes.DWORD),
            ('rcMonitor', wintypes.RECT),
            ('rcWork', wintypes.RECT),
            ('dwFlags', wintypes.DWORD),
            ('szDevice', ctypes.c_wchar * 32),
        ]

    monitors: List[dict] = []

    @ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    def _enum_proc(hmon, _hdc, _lprect, _lparam):
        info = MONITORINFOEX()
        info.cbSize = ctypes.sizeof(MONITORINFOEX)
        try:
            ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        except Exception:
            return 1
        rect = info.rcMonitor
        scale_factor = 1.0
        try:
            # MDT_EFFECTIVE_DPI = 0 (Win 8.1+); falls back to 96 if unavailable
            dpi_x = ctypes.c_uint(96)
            dpi_y = ctypes.c_uint(96)
            ctypes.windll.shcore.GetDpiForMonitor(
                hmon, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
            scale_factor = dpi_x.value / 96.0
        except (AttributeError, OSError):
            pass
        monitors.append({
            'idx': len(monitors),
            'rect': (
                rect.left, rect.top,
                rect.right - rect.left, rect.bottom - rect.top,
            ),
            'work_rect': (
                info.rcWork.left, info.rcWork.top,
                info.rcWork.right - info.rcWork.left,
                info.rcWork.bottom - info.rcWork.top,
            ),
            'scale_factor': scale_factor,
            'is_primary': bool(info.dwFlags & MONITORINFOF_PRIMARY),
            'name': info.szDevice,
        })
        return 1

    try:
        ctypes.windll.user32.EnumDisplayMonitors(0, 0, _enum_proc, 0)
    except Exception as e:
        logger.debug(f"EnumDisplayMonitors failed: {e}")
    return monitors


def capture_window_one_shot(hwnd: int, *, fmt: str = 'jpeg',
                            quality: int = 70) -> Optional[bytes]:
    """Capture a single window's pixels even when it's occluded /
    not the foreground.

    Uses ``user32.PrintWindow`` with ``PW_RENDERFULLCONTENT = 0x02``
    which captures DWM-rendered content correctly for windows that
    don't respond to ``WM_PRINT`` (most modern Win10+ apps including
    Chrome / Edge / UWP).  Falls back to plain ``PrintWindow`` (flag
    = 0) for older Windows where the flag is unsupported.

    Args:
        hwnd: Window handle from :func:`list_windows`.
        fmt: 'jpeg' (default) or 'png'.
        quality: JPEG quality 1–100 (ignored for png).

    Returns:
        Image bytes, or None if the window is gone / dimensions zero /
        capture failed.

    Failure modes (callers should handle):
      * DRM-protected / cloaked windows return all-black pixels.  Check
        ``WindowInfo.is_protected`` before relying on the capture.
      * Pre-Win 10 1903 lacks PW_RENDERFULLCONTENT.  This function
        downgrades to flag=0 with a debug log.
      * Window minimized: returns last-saved DWM thumbnail (may be stale).
    """
    if platform.system() != 'Windows' or not _win32gui or not _PIL_Image:
        return None
    try:
        import ctypes
    except ImportError:
        return None
    if not _win32gui.IsWindow(hwnd):
        return None
    try:
        left, top, right, bottom = _win32gui.GetClientRect(hwnd)
    except Exception:
        return None
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None

    PW_RENDERFULLCONTENT = 0x02

    hwnd_dc = _win32gui.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    mfc_dc = None
    save_dc = None
    bitmap = None
    try:
        mfc_dc = _win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = _win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)
        result = ctypes.windll.user32.PrintWindow(
            hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        if not result:
            # Older Win — retry without the flag.
            result = ctypes.windll.user32.PrintWindow(
                hwnd, save_dc.GetSafeHdc(), 0)
        if not result:
            return None
        bmp_info = bitmap.GetInfo()
        bmp_data = bitmap.GetBitmapBits(True)
        img = _PIL_Image.frombuffer(
            'RGB',
            (bmp_info['bmWidth'], bmp_info['bmHeight']),
            bmp_data, 'raw', 'BGRX', 0, 1,
        )
        buf = io.BytesIO()
        if fmt.lower() == 'png':
            img.save(buf, format='PNG', optimize=True)
        else:
            img.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"capture_window_one_shot failed for hwnd={hwnd}: {e}")
        return None
    finally:
        try:
            if bitmap is not None:
                _win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc is not None:
                save_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            _win32gui.ReleaseDC(hwnd, hwnd_dc)
        except Exception:
            pass


def list_windows(*, include_minimized: bool = False) -> List[dict]:
    """VLM-friendly thin wrapper: return list of dicts (not WindowInfo
    objects) ready to ship to the VLM grounding prompt.

    Calls into :class:`WindowEnumerator` so there's one canonical
    enumerator implementation for the whole codebase.
    """
    enum = WindowEnumerator()
    return [w.to_dict() for w in enum.list_windows(
        include_minimized=include_minimized)]
