"""
build_compiled_wheel.py — Build source-protected wheels for HARTOS/HevolveAI

Produces pip-installable wheels with .pyc only (no .py source).
Used for distributing proprietary backend code alongside open-source Nunba.

Usage:
    python scripts/build_compiled_wheel.py          # Build HARTOS wheel
    python scripts/build_compiled_wheel.py --all    # Build HARTOS + HevolveAI

Output:
    dist/hart_backend-X.Y.Z-cpXX-cpXX-platform.whl  (compiled-only)
"""
import os
import sys
import shutil
import tempfile
import subprocess
import compileall
import zipfile
import glob
import re
from pathlib import Path


def compile_wheel(project_dir, output_dir="dist"):
    """Build a wheel, then strip .py source leaving only .pyc."""
    project_dir = os.path.abspath(project_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building wheel for {os.path.basename(project_dir)}...")

    # Step 1: Build a normal wheel
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", project_dir,
             "--no-deps", "--wheel-dir", tmpdir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"ERROR: wheel build failed:\n{result.stderr}")
            return None

        # Find the built wheel
        wheels = glob.glob(os.path.join(tmpdir, "*.whl"))
        if not wheels:
            print("ERROR: no wheel produced")
            return None

        src_wheel = wheels[0]
        wheel_name = os.path.basename(src_wheel)
        print(f"  Built: {wheel_name}")

        # Step 2: Repack wheel with .pyc only
        compiled_wheel = os.path.join(output_dir, wheel_name)
        _repack_compiled(src_wheel, compiled_wheel)
        print(f"  Compiled: {compiled_wheel}")
        return compiled_wheel


def _repack_compiled(src_wheel, dst_wheel):
    """Repack a wheel replacing .py files with compiled .pyc."""
    with tempfile.TemporaryDirectory() as extract_dir:
        # Extract
        with zipfile.ZipFile(src_wheel, 'r') as zf:
            zf.extractall(extract_dir)

        # Compile all .py files
        compileall.compile_dir(
            extract_dir, force=True, quiet=2,
            optimize=2,  # -OO: strip docstrings + asserts
        )

        # Walk and replace .py with .pyc
        py_removed = 0
        pyc_kept = 0
        for root, dirs, files in os.walk(extract_dir):
            for fname in files:
                fpath = os.path.join(root, fname)

                if fname.endswith('.py') and fname != '__init__.py':
                    # Check if .pyc exists in __pycache__
                    pycache = os.path.join(root, '__pycache__')
                    base = fname[:-3]
                    pyc_candidates = glob.glob(
                        os.path.join(pycache, f"{base}.cpython-*.opt-2.pyc")
                    ) or glob.glob(
                        os.path.join(pycache, f"{base}.cpython-*.pyc")
                    )

                    if pyc_candidates:
                        # Move .pyc to replace .py (rename to .pyc alongside)
                        pyc_src = pyc_candidates[0]
                        pyc_dst = fpath + 'c'  # module.pyc next to where module.py was
                        shutil.copy2(pyc_src, pyc_dst)
                        os.remove(fpath)
                        py_removed += 1
                        pyc_kept += 1
                    # else: keep .py (compilation failed, rare)

                elif fname == '__init__.py':
                    # Keep __init__.py but make it minimal (just declares package)
                    # This is required for Python package discovery
                    pycache = os.path.join(root, '__pycache__')
                    base = '__init__'
                    pyc_candidates = glob.glob(
                        os.path.join(pycache, f"{base}.cpython-*.opt-2.pyc")
                    ) or glob.glob(
                        os.path.join(pycache, f"{base}.cpython-*.pyc")
                    )
                    if pyc_candidates:
                        pyc_src = pyc_candidates[0]
                        pyc_dst = fpath + 'c'
                        shutil.copy2(pyc_src, pyc_dst)
                        # Replace __init__.py with stub
                        with open(fpath, 'w') as f:
                            f.write("# Compiled package\n")
                        pyc_kept += 1

        # Remove __pycache__ dirs (pyc files already moved)
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            if os.path.basename(root) == '__pycache__':
                shutil.rmtree(root, ignore_errors=True)

        # Update RECORD in dist-info (wheel integrity file)
        for root, dirs, files in os.walk(extract_dir):
            if root.endswith('.dist-info'):
                record_path = os.path.join(root, 'RECORD')
                if os.path.isfile(record_path):
                    _update_record(extract_dir, record_path)

        # Repack as wheel (zip)
        with zipfile.ZipFile(dst_wheel, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(extract_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, extract_dir)
                    zf.write(fpath, arcname)

        print(f"  Stripped {py_removed} .py files, kept {pyc_kept} .pyc files")


def _update_record(base_dir, record_path):
    """Regenerate RECORD file after modifying wheel contents."""
    import hashlib
    import base64

    lines = []
    dist_info_rel = os.path.relpath(os.path.dirname(record_path), base_dir)

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, base_dir).replace('\\', '/')

            if rel.endswith('/RECORD'):
                # RECORD itself has no hash
                lines.append(f"{rel},,")
                continue

            with open(fpath, 'rb') as f:
                data = f.read()
            digest = hashlib.sha256(data).digest()
            b64 = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
            lines.append(f"{rel},sha256={b64},{len(data)}")

    with open(record_path, 'w', newline='\n') as f:
        f.write('\n'.join(sorted(lines)) + '\n')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hartos_dir = os.path.dirname(script_dir)  # HARTOS project root

    # Build HARTOS (hart-backend)
    wheel = compile_wheel(hartos_dir)
    if wheel:
        print(f"\nHARTOS compiled wheel: {wheel}")

    # Build HevolveAI if --all
    if '--all' in sys.argv:
        hevolveai_dir = os.path.join(os.path.dirname(hartos_dir), 'hevolveai')
        if os.path.isdir(hevolveai_dir):
            wheel2 = compile_wheel(hevolveai_dir)
            if wheel2:
                print(f"HevolveAI compiled wheel: {wheel2}")
        else:
            print(f"HevolveAI not found at {hevolveai_dir}")

    print("\nDone. Install with: pip install dist/*.whl")
