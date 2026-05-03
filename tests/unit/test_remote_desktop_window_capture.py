"""
Tests for window-level capture + multi-session (tab detach).

Covers: window_capture.py, window_session.py
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestWindowInfo(unittest.TestCase):
    """Tests for WindowInfo dataclass."""

    def test_window_info_creation(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        w = WindowInfo(
            hwnd=12345, title='Notepad', process_name='notepad.exe',
            pid=1000, rect=(0, 0, 800, 600), visible=True, minimized=False,
        )
        self.assertEqual(w.hwnd, 12345)
        self.assertEqual(w.title, 'Notepad')
        self.assertEqual(w.process_name, 'notepad.exe')
        self.assertTrue(w.visible)

    def test_window_info_to_dict(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        w = WindowInfo(
            hwnd=1, title='Test', process_name='test.exe',
            pid=42, rect=(10, 20, 300, 200), visible=True, minimized=False,
        )
        d = w.to_dict()
        self.assertEqual(d['hwnd'], 1)
        self.assertEqual(d['title'], 'Test')
        self.assertEqual(list(d['rect']), [10, 20, 300, 200])

    def test_window_info_from_dict(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        d = {
            'hwnd': 99, 'title': 'CMD', 'process_name': 'cmd.exe',
            'pid': 500, 'rect': [0, 0, 640, 480],
            'visible': True, 'minimized': False,
        }
        w = WindowInfo.from_dict(d)
        self.assertEqual(w.hwnd, 99)
        self.assertEqual(w.title, 'CMD')

    def test_window_info_defaults(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        w = WindowInfo(
            hwnd=0, title='', process_name='',
            pid=0, rect=(0, 0, 0, 0),
        )
        self.assertTrue(w.visible)  # default
        self.assertFalse(w.minimized)  # default


class TestWindowEnumerator(unittest.TestCase):
    """Tests for WindowEnumerator."""

    def test_enumerator_creation(self):
        from integrations.remote_desktop.window_capture import WindowEnumerator
        e = WindowEnumerator()
        self.assertIsNotNone(e)

    def test_list_windows_returns_list(self):
        from integrations.remote_desktop.window_capture import WindowEnumerator
        e = WindowEnumerator()
        result = e.list_windows()
        self.assertIsInstance(result, list)

    @patch('platform.system', return_value='Linux')
    def test_list_windows_linux_fallback(self, mock_sys):
        from integrations.remote_desktop.window_capture import WindowEnumerator
        e = WindowEnumerator()
        # Should return empty list on Linux without X11
        result = e.list_windows()
        self.assertIsInstance(result, list)

    def test_get_window_by_title_not_found(self):
        from integrations.remote_desktop.window_capture import WindowEnumerator
        e = WindowEnumerator()
        # Search for nonexistent window
        result = e.get_window_by_title('NONEXISTENT_WINDOW_XYZ_12345')
        self.assertIsNone(result)

    def test_get_window_by_pid_not_found(self):
        from integrations.remote_desktop.window_capture import WindowEnumerator
        e = WindowEnumerator()
        result = e.get_window_by_pid(999999999)
        self.assertIsNone(result)

    def test_get_window_by_title_case_insensitive(self):
        """Title matching should be case-insensitive."""
        from integrations.remote_desktop.window_capture import (
            WindowEnumerator, WindowInfo,
        )
        e = WindowEnumerator()
        # Mock list_windows to return a test window
        e.list_windows = lambda **kw: [
            WindowInfo(hwnd=1, title='My Notepad', process_name='notepad.exe',
                       pid=100, rect=(0, 0, 800, 600)),
        ]
        result = e.get_window_by_title('notepad')
        self.assertIsNotNone(result)
        self.assertEqual(result.hwnd, 1)


class TestWindowCaptureConfig(unittest.TestCase):
    """Tests for WindowCaptureConfig."""

    def test_default_config(self):
        from integrations.remote_desktop.window_capture import WindowCaptureConfig
        cfg = WindowCaptureConfig()
        self.assertGreater(cfg.quality, 0)
        self.assertGreater(cfg.max_fps, 0)

    def test_custom_config(self):
        from integrations.remote_desktop.window_capture import WindowCaptureConfig
        cfg = WindowCaptureConfig(quality=50, max_fps=15, scale_factor=0.5)
        self.assertEqual(cfg.quality, 50)
        self.assertEqual(cfg.max_fps, 15)
        self.assertEqual(cfg.scale_factor, 0.5)


class TestWindowCapture(unittest.TestCase):
    """Tests for WindowCapture."""

    def _make_window_info(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        return WindowInfo(
            hwnd=12345, title='Test', process_name='test.exe',
            pid=100, rect=(0, 0, 800, 600),
        )

    def test_window_capture_creation(self):
        from integrations.remote_desktop.window_capture import WindowCapture
        winfo = self._make_window_info()
        cap = WindowCapture(winfo)
        self.assertIsNotNone(cap)

    def test_capture_frame_returns_bytes_or_none(self):
        from integrations.remote_desktop.window_capture import WindowCapture
        winfo = self._make_window_info()
        cap = WindowCapture(winfo)
        frame = cap.capture_frame()
        # May return None if no display/capture backend available
        self.assertTrue(frame is None or isinstance(frame, bytes))

    def test_get_window_info(self):
        from integrations.remote_desktop.window_capture import WindowCapture
        winfo = self._make_window_info()
        cap = WindowCapture(winfo)
        result = cap.get_window_info()
        self.assertEqual(result.hwnd, 12345)

    def test_window_capture_with_config(self):
        from integrations.remote_desktop.window_capture import (
            WindowCapture, WindowCaptureConfig,
        )
        winfo = self._make_window_info()
        cfg = WindowCaptureConfig(quality=30, max_fps=10)
        cap = WindowCapture(winfo, config=cfg)
        self.assertIsNotNone(cap)


class TestWindowSession(unittest.TestCase):
    """Tests for WindowSession dataclass."""

    def test_window_session_creation(self):
        from integrations.remote_desktop.window_session import WindowSession
        s = WindowSession(
            session_id='ws-001',
            window_hwnd=12345,
            window_title='Notepad',
            process_name='notepad.exe',
            started_at=1000.0,
        )
        self.assertEqual(s.session_id, 'ws-001')
        self.assertEqual(s.window_hwnd, 12345)

    def test_window_session_to_dict(self):
        from integrations.remote_desktop.window_session import WindowSession
        s = WindowSession(
            session_id='ws-002',
            window_hwnd=1,
            window_title='CMD',
            process_name='cmd.exe',
            started_at=2000.0,
        )
        d = s.to_dict()
        self.assertEqual(d['session_id'], 'ws-002')
        self.assertIn('window_title', d)


class TestWindowSessionManager(unittest.TestCase):
    """Tests for WindowSessionManager."""

    def test_manager_creation(self):
        from integrations.remote_desktop.window_session import WindowSessionManager
        m = WindowSessionManager()
        self.assertIsNotNone(m)

    def test_list_available_windows(self):
        from integrations.remote_desktop.window_session import WindowSessionManager
        m = WindowSessionManager()
        result = m.list_available_windows()
        self.assertIsInstance(result, list)

    def test_get_active_sessions_empty(self):
        from integrations.remote_desktop.window_session import WindowSessionManager
        m = WindowSessionManager()
        sessions = m.get_active_window_sessions()
        self.assertIsInstance(sessions, list)
        self.assertEqual(len(sessions), 0)

    def test_stop_nonexistent_session(self):
        from integrations.remote_desktop.window_session import WindowSessionManager
        m = WindowSessionManager()
        result = m.stop_window_session('nonexistent-id')
        self.assertFalse(result)

    def test_stop_all_empty(self):
        from integrations.remote_desktop.window_session import WindowSessionManager
        m = WindowSessionManager()
        m.stop_all()  # Should not raise

    def test_singleton(self):
        import integrations.remote_desktop.window_session as ws_mod
        ws_mod._window_session_manager = None
        m1 = ws_mod.get_window_session_manager()
        m2 = ws_mod.get_window_session_manager()
        self.assertIs(m1, m2)
        ws_mod._window_session_manager = None  # cleanup


class TestWindowOrchestratorIntegration(unittest.TestCase):
    """Tests for window methods on orchestrator."""

    def test_list_remote_windows(self):
        from integrations.remote_desktop.orchestrator import RemoteDesktopOrchestrator
        orch = RemoteDesktopOrchestrator()
        result = orch.list_remote_windows()
        self.assertIsInstance(result, list)

    def test_get_window_sessions_empty(self):
        from integrations.remote_desktop.orchestrator import RemoteDesktopOrchestrator
        orch = RemoteDesktopOrchestrator()
        result = orch.get_window_sessions()
        self.assertIsInstance(result, list)

    def test_stop_window_stream_nonexistent(self):
        from integrations.remote_desktop.orchestrator import RemoteDesktopOrchestrator
        orch = RemoteDesktopOrchestrator()
        result = orch.stop_window_stream('fake-id')
        self.assertFalse(result)


# ════════════════════════════════════════════════════════════════════
# Phase 1 of memory/vlm_best_of_all_worlds_plan.md §1 —
# occlusion detection + multi-monitor enumeration + PrintWindow
# capture.  Pure-logic tests (occlusion + monitor assignment) need
# no OS hooks; OS-bound tests (list_monitors, capture_window_one_shot)
# skip on non-Windows.
# ════════════════════════════════════════════════════════════════════


class TestComputeOcclusion(unittest.TestCase):
    """_compute_occlusion is pure: takes a top-to-bottom-z-order list
    and annotates is_occluded / occluded_pct in place."""

    def _win(self, hwnd, rect, minimized=False):
        from integrations.remote_desktop.window_capture import WindowInfo
        return WindowInfo(
            hwnd=hwnd, title=f'w{hwnd}', process_name='', pid=0,
            rect=rect, visible=True, minimized=minimized)

    def test_top_window_never_occluded(self):
        """The first window in z-order has nothing above it."""
        from integrations.remote_desktop.window_capture import _compute_occlusion
        windows = [
            self._win(1, (0, 0, 100, 100)),
            self._win(2, (0, 0, 100, 100)),
        ]
        _compute_occlusion(windows)
        self.assertFalse(windows[0].is_occluded)
        self.assertEqual(windows[0].occluded_pct, 0.0)

    def test_fully_covered_window_is_100_pct_occluded(self):
        from integrations.remote_desktop.window_capture import _compute_occlusion
        # Top window covers the entire bottom one.
        windows = [
            self._win(1, (0, 0, 100, 100)),
            self._win(2, (0, 0, 100, 100)),
        ]
        _compute_occlusion(windows)
        self.assertTrue(windows[1].is_occluded)
        self.assertAlmostEqual(windows[1].occluded_pct, 100.0)

    def test_half_covered_window_reports_50_pct(self):
        from integrations.remote_desktop.window_capture import _compute_occlusion
        # Top covers right half of bottom.
        windows = [
            self._win(1, (50, 0, 50, 100)),  # right half
            self._win(2, (0, 0, 100, 100)),
        ]
        _compute_occlusion(windows)
        self.assertAlmostEqual(windows[1].occluded_pct, 50.0)
        self.assertTrue(windows[1].is_occluded)

    def test_small_overlay_under_5pct_not_marked_occluded(self):
        """Tray badges / notification bubbles shouldn't count as
        occluding the window underneath."""
        from integrations.remote_desktop.window_capture import _compute_occlusion
        # Top covers a 10x10 corner (1% of 100x100 bottom).
        windows = [
            self._win(1, (0, 0, 10, 10)),
            self._win(2, (0, 0, 100, 100)),
        ]
        _compute_occlusion(windows)
        self.assertAlmostEqual(windows[1].occluded_pct, 1.0)
        self.assertFalse(windows[1].is_occluded)

    def test_no_overlap_means_not_occluded(self):
        from integrations.remote_desktop.window_capture import _compute_occlusion
        windows = [
            self._win(1, (0, 0, 50, 50)),     # top-left
            self._win(2, (60, 60, 50, 50)),   # bottom-right, no overlap
        ]
        _compute_occlusion(windows)
        self.assertFalse(windows[1].is_occluded)
        self.assertEqual(windows[1].occluded_pct, 0.0)

    def test_two_overlapping_covers_capped_at_100(self):
        """Multiple covers on the same pixels must not double-count
        (would make occluded_pct > 100 nonsensically)."""
        from integrations.remote_desktop.window_capture import _compute_occlusion
        windows = [
            self._win(1, (0, 0, 100, 100)),  # full cover
            self._win(2, (0, 0, 100, 100)),  # same — but already counted
            self._win(3, (0, 0, 100, 100)),  # bottom
        ]
        _compute_occlusion(windows)
        self.assertLessEqual(windows[2].occluded_pct, 100.0)

    def test_minimized_window_skipped_in_overlap(self):
        """A minimized window above shouldn't count as occluding
        anything (it's in the taskbar, not on screen)."""
        from integrations.remote_desktop.window_capture import _compute_occlusion
        windows = [
            self._win(1, (0, 0, 100, 100), minimized=True),
            self._win(2, (0, 0, 100, 100)),
        ]
        _compute_occlusion(windows)
        self.assertEqual(windows[1].occluded_pct, 0.0)

    def test_zero_size_window_skipped(self):
        from integrations.remote_desktop.window_capture import _compute_occlusion
        windows = [self._win(1, (0, 0, 0, 0))]
        _compute_occlusion(windows)  # must not raise / div-by-zero
        self.assertFalse(windows[0].is_occluded)


class TestAssignMonitors(unittest.TestCase):

    def _win(self, rect):
        from integrations.remote_desktop.window_capture import WindowInfo
        return WindowInfo(hwnd=1, title='', process_name='',
                          pid=0, rect=rect)

    def test_window_in_primary_monitor(self):
        from integrations.remote_desktop.window_capture import _assign_monitors
        monitors = [
            {'idx': 0, 'rect': (0, 0, 1920, 1080)},
            {'idx': 1, 'rect': (1920, 0, 1920, 1080)},
        ]
        windows = [self._win((100, 100, 800, 600))]  # center=(500,400) → primary
        _assign_monitors(windows, monitors)
        self.assertEqual(windows[0].monitor_idx, 0)

    def test_window_in_second_monitor(self):
        from integrations.remote_desktop.window_capture import _assign_monitors
        monitors = [
            {'idx': 0, 'rect': (0, 0, 1920, 1080)},
            {'idx': 1, 'rect': (1920, 0, 1920, 1080)},
        ]
        # Center at (2500, 500) — clearly in monitor 1
        windows = [self._win((2100, 100, 800, 800))]
        _assign_monitors(windows, monitors)
        self.assertEqual(windows[0].monitor_idx, 1)

    def test_window_off_all_monitors_gets_minus_1(self):
        from integrations.remote_desktop.window_capture import _assign_monitors
        monitors = [{'idx': 0, 'rect': (0, 0, 1920, 1080)}]
        windows = [self._win((5000, 5000, 100, 100))]
        _assign_monitors(windows, monitors)
        self.assertEqual(windows[0].monitor_idx, -1)


class TestWindowInfoNewFieldsBackCompat(unittest.TestCase):
    """The new Phase-1 fields all have defaults — existing callers
    that build WindowInfo without them must keep working."""

    def test_default_values_safe(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        w = WindowInfo(hwnd=1, title='t', process_name='p',
                       pid=42, rect=(0, 0, 100, 100))
        self.assertEqual(w.z_order, 0)
        self.assertFalse(w.is_foreground)
        self.assertFalse(w.is_occluded)
        self.assertEqual(w.occluded_pct, 0.0)
        self.assertFalse(w.is_protected)
        self.assertEqual(w.monitor_idx, -1)

    def test_to_dict_includes_new_fields(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        w = WindowInfo(hwnd=1, title='t', process_name='p',
                       pid=42, rect=(0, 0, 100, 100),
                       z_order=3, is_foreground=True,
                       is_occluded=True, occluded_pct=42.7,
                       is_protected=True, monitor_idx=1)
        d = w.to_dict()
        self.assertEqual(d['z_order'], 3)
        self.assertTrue(d['is_foreground'])
        self.assertTrue(d['is_occluded'])
        self.assertEqual(d['occluded_pct'], 42.7)
        self.assertTrue(d['is_protected'])
        self.assertEqual(d['monitor_idx'], 1)

    def test_from_dict_round_trip(self):
        from integrations.remote_desktop.window_capture import WindowInfo
        d = {'hwnd': 1, 'title': 't', 'process_name': 'p',
             'pid': 42, 'rect': [0, 0, 100, 100],
             'z_order': 5, 'is_foreground': True,
             'occluded_pct': 33.3, 'monitor_idx': 2}
        w = WindowInfo.from_dict(d)
        self.assertEqual(w.z_order, 5)
        self.assertTrue(w.is_foreground)
        self.assertEqual(w.occluded_pct, 33.3)
        self.assertEqual(w.monitor_idx, 2)
        # Missing fields fall back to defaults
        self.assertFalse(w.is_protected)

    def test_legacy_dict_without_new_fields_still_works(self):
        """Old serialized payloads (e.g. cached in DB) must deserialize
        without raising even though new fields aren't present."""
        from integrations.remote_desktop.window_capture import WindowInfo
        d = {'hwnd': 1, 'title': 'old', 'process_name': '',
             'pid': 0, 'rect': [0, 0, 100, 100]}
        w = WindowInfo.from_dict(d)
        self.assertEqual(w.z_order, 0)
        self.assertFalse(w.is_protected)


@unittest.skipUnless(__import__('platform').system() == 'Windows',
                     'list_monitors() Windows backend; mac/linux in Phase 2')
class TestListMonitorsWindows(unittest.TestCase):
    """list_monitors must return at least one monitor on a real
    Windows host with at least one display attached."""

    def test_returns_at_least_one_monitor(self):
        from integrations.remote_desktop.window_capture import list_monitors
        monitors = list_monitors()
        self.assertGreaterEqual(len(monitors), 1)

    def test_monitor_dict_shape(self):
        from integrations.remote_desktop.window_capture import list_monitors
        monitors = list_monitors()
        if not monitors:
            self.skipTest('no displays detected — possibly headless CI')
        m = monitors[0]
        for key in ('idx', 'rect', 'scale_factor', 'is_primary', 'name'):
            self.assertIn(key, m, f'list_monitors entry missing "{key}"')
        # rect = (x, y, w, h)
        self.assertEqual(len(m['rect']), 4)
        self.assertGreater(m['rect'][2], 0, 'monitor width should be positive')
        self.assertGreater(m['rect'][3], 0, 'monitor height should be positive')

    def test_exactly_one_primary(self):
        from integrations.remote_desktop.window_capture import list_monitors
        monitors = list_monitors()
        if not monitors:
            self.skipTest('no displays detected')
        primaries = [m for m in monitors if m['is_primary']]
        self.assertEqual(len(primaries), 1,
                         'there should be exactly one primary monitor')


class TestListMonitorsNonWindows(unittest.TestCase):
    """Phase-2 will add macOS/Linux backends.  Until then the function
    must return an empty list (not raise) so callers behave correctly."""

    def test_empty_on_non_windows(self):
        with patch('integrations.remote_desktop.window_capture.platform.system',
                   return_value='Darwin'):
            from integrations.remote_desktop.window_capture import list_monitors
            self.assertEqual(list_monitors(), [])


class TestListWindowsModuleWrapper(unittest.TestCase):
    """list_windows() module function returns dicts (not WindowInfo)
    so VLM callers can serialize directly to JSON for the prompt."""

    def test_returns_list_of_dicts_with_new_fields(self):
        """When the underlying enumerator yields a WindowInfo with
        Phase-1 fields populated, the wrapper preserves them."""
        from integrations.remote_desktop.window_capture import (
            list_windows, WindowInfo)

        fake_win = WindowInfo(
            hwnd=42, title='Notepad', process_name='notepad.exe',
            pid=1234, rect=(100, 100, 800, 600),
            z_order=2, is_foreground=False,
            is_occluded=True, occluded_pct=37.5,
            is_protected=False, monitor_idx=0)

        with patch('integrations.remote_desktop.window_capture.WindowEnumerator') \
                as mock_enum:
            mock_enum.return_value.list_windows.return_value = [fake_win]
            result = list_windows()

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(result[0]['title'], 'Notepad')
        self.assertEqual(result[0]['z_order'], 2)
        self.assertTrue(result[0]['is_occluded'])
        self.assertEqual(result[0]['occluded_pct'], 37.5)
        self.assertEqual(result[0]['monitor_idx'], 0)


class TestCaptureWindowOneShot(unittest.TestCase):
    """capture_window_one_shot fails gracefully on bad inputs and
    returns None on non-Windows (Phase 2 will add backends).  Real
    capture is exercised against the test process's own window in
    the integration tier — too flaky for unit suite."""

    def test_returns_none_for_invalid_hwnd_on_non_windows(self):
        with patch('integrations.remote_desktop.window_capture.platform.system',
                   return_value='Linux'):
            from integrations.remote_desktop.window_capture import capture_window_one_shot
            self.assertIsNone(capture_window_one_shot(0))

    @unittest.skipUnless(__import__('platform').system() == 'Windows',
                         'PrintWindow Windows-only; Phase 2 covers mac/linux')
    def test_returns_none_for_invalid_hwnd(self):
        from integrations.remote_desktop.window_capture import capture_window_one_shot
        # 0 is never a valid HWND.
        self.assertIsNone(capture_window_one_shot(0))


if __name__ == '__main__':
    unittest.main()
