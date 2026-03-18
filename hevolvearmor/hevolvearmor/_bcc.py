"""
BCC Mode — Compile Python modules to C extensions via Cython.

Irreversible protection: no bytecode to decompile. The compiled .so/.pyd
contains native machine instructions only.

Usage (build-time):
    from hevolvearmor._bcc import compile_package_to_c
    stats = compile_package_to_c(
        source_dir='./hevolveai',
        output_dir='./hevolveai_compiled',
        patterns=['embodied_ai.core.*', 'embodied_ai.memory.*'],
        skip_patterns=['embodied_ai.inference.*'],  # hot paths, torch interop
    )

The compiled extensions are then encrypted with AES-256-GCM (double protection:
native code + encryption at rest).

Requires: Cython, a C compiler (gcc/clang/MSVC)
"""
import fnmatch
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

_SKIP_DIRS = frozenset({
    '__pycache__', '.git', 'tests', 'test', 'legacy', 'dashboard',
    '.egg-info', 'dist', 'build',
})

# Modules that use heavy C extension interop — Cython compilation
# may break their runtime behavior
_DEFAULT_SKIP = [
    '*.inference.*',     # torch tensor operations
    '*.models.qwen_*',  # Qwen VLM, heavy torch
    '*.learning.lora_*', # LoRA adapters, torch
    '*.rl_ef.*',         # reinforcement learning, torch
]


def _get_ext_suffix():
    """Get platform extension suffix (e.g. .cp311-win_amd64.pyd)."""
    return sysconfig.get_config_var('EXT_SUFFIX') or '.so'


def _cythonize_file(py_path: str, output_dir: str, rel_path: str,
                    verbose: bool = True) -> bool:
    """Compile a single .py file to .c then to .so/.pyd.

    Returns True on success.
    """
    ext_suffix = _get_ext_suffix()
    module_name = os.path.splitext(os.path.basename(py_path))[0]

    # Output paths
    c_path = os.path.join(output_dir, rel_path.replace('.py', '.c'))
    ext_rel = rel_path.replace('.py', ext_suffix)
    ext_path = os.path.join(output_dir, ext_rel)

    os.makedirs(os.path.dirname(c_path), exist_ok=True)
    os.makedirs(os.path.dirname(ext_path), exist_ok=True)

    # Step 1: Cython .py → .c
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'cython', '-3', '--embed-positions',
             '-o', c_path, py_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            if verbose:
                print(f'  [BCC-FAIL] {rel_path}: Cython error: {result.stderr[:200]}')
            return False
    except FileNotFoundError:
        if verbose:
            print(f'  [BCC-FAIL] Cython not found')
        return False
    except subprocess.TimeoutExpired:
        if verbose:
            print(f'  [BCC-FAIL] {rel_path}: Cython timeout')
        return False

    # Step 2: Compile .c → .so/.pyd
    include_dir = sysconfig.get_path('include')
    lib_dir = sysconfig.get_config_var('LIBDIR') or ''

    if sys.platform == 'win32':
        # MSVC or MinGW
        python_lib = os.path.join(
            os.path.dirname(sys.executable), 'libs',
            f'python{sys.version_info.major}{sys.version_info.minor}.lib'
        )
        compile_cmd = [
            sys.executable, '-c',
            f"""
import setuptools
from Cython.Build import cythonize
from setuptools import Extension, setup
import sys, os
ext = Extension(
    '{module_name}',
    sources=[r'{c_path}'],
    include_dirs=[r'{include_dir}'],
)
setup(
    ext_modules=[ext],
    script_args=['build_ext', '--inplace',
                 '--build-lib', r'{os.path.dirname(ext_path)}',
                 '--build-temp', r'{tempfile.mkdtemp()}'],
)
"""
        ]
    else:
        # GCC/Clang
        compile_cmd = [
            'gcc' if shutil.which('gcc') else 'cc',
            '-shared', '-fPIC', '-O2',
            f'-I{include_dir}',
            '-o', ext_path,
            c_path,
        ]
        if lib_dir:
            compile_cmd.insert(-1, f'-L{lib_dir}')
        # Link against Python
        py_ver = f'{sys.version_info.major}.{sys.version_info.minor}'
        compile_cmd.extend([f'-lpython{py_ver}'])

    try:
        result = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            if verbose:
                err = result.stderr[:300] if result.stderr else result.stdout[:300]
                print(f'  [BCC-FAIL] {rel_path}: C compile error: {err}')
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        if verbose:
            print(f'  [BCC-FAIL] {rel_path}: {type(e).__name__}')
        return False

    # Clean up .c file (we only ship the .so/.pyd)
    try:
        os.remove(c_path)
    except OSError:
        pass

    if verbose:
        size = os.path.getsize(ext_path) if os.path.isfile(ext_path) else 0
        print(f'  [BCC-OK] {rel_path} -> {ext_rel} ({size:,} bytes)')

    return os.path.isfile(ext_path)


def compile_package_to_c(
    source_dir: str,
    output_dir: str,
    patterns: list = None,
    skip_patterns: list = None,
    verbose: bool = True,
) -> dict:
    """Compile a Python package tree to C extensions via Cython.

    Args:
        source_dir: path to package root
        output_dir: output directory for compiled extensions
        patterns: fnmatch patterns of modules to compile (None = all)
        skip_patterns: fnmatch patterns to skip (default: torch/numpy interop)
        verbose: print progress

    Returns:
        dict with {compiled, failed, skipped}
    """
    skip_patterns = skip_patterns or _DEFAULT_SKIP

    stats = {'compiled': 0, 'failed': 0, 'skipped': 0}

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(source_dir):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.endswith('.egg-info')]

        for fname in sorted(filenames):
            if not fname.endswith('.py'):
                continue

            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, source_dir).replace(os.sep, '/')
            dotted = rel_path.replace('/', '.').replace('.py', '')

            # __init__.py must remain as .py (package marker)
            if fname == '__init__.py':
                dst = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # Copy a minimal __init__.py that imports from the compiled extension
                shutil.copy2(full_path, dst)
                stats['skipped'] += 1
                continue

            # Check patterns
            if patterns and not any(fnmatch.fnmatch(dotted, p) for p in patterns):
                stats['skipped'] += 1
                continue

            if skip_patterns and any(fnmatch.fnmatch(dotted, p) for p in skip_patterns):
                # Copy source as-is (will be encrypted later by the main pipeline)
                dst = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(full_path, dst)
                stats['skipped'] += 1
                if verbose:
                    print(f'  [BCC-SKIP] {rel_path} (matches skip pattern)')
                continue

            # Compile
            if _cythonize_file(full_path, output_dir, rel_path, verbose):
                stats['compiled'] += 1
            else:
                # Fallback: copy source (will be encrypted by main pipeline)
                dst = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(full_path, dst)
                stats['failed'] += 1

    if verbose:
        print(f'\n  BCC: {stats["compiled"]} compiled, '
              f'{stats["failed"]} failed (fallback to .py), '
              f'{stats["skipped"]} skipped')

    return stats
