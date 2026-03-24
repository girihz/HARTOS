"""
Build-time encryption of Python packages with optional transforms.

Pipeline: .py → [RFT rename] → [assert-import] → [private mode] →
          compile → [string encrypt] → [per-function wrap] →
          marshal → AES-256-GCM encrypt → .enc
"""
import importlib.util
import marshal
import os
import shutil
import struct
import sys

from hevolvearmor._transforms import TransformConfig, apply_transforms

_SKIP_DIRS = frozenset({
    '__pycache__', '.git', '.tox', 'tests', 'test', 'legacy',
    'dashboard', '.egg-info', 'dist', 'build', '.mypy_cache',
    '.pytest_cache', '.ruff_cache',
})


def compile_to_pyc_bytes(source_path: str,
                          config: TransformConfig = None,
                          encrypt_fn=None) -> bytes:
    """Compile a .py file to .pyc bytes with optional transforms.

    Args:
        source_path: path to .py file
        config: transform config (None = no transforms, just compile)
        encrypt_fn: Rust encrypt function for string/function wrapping

    Returns:
        .pyc bytes (header + marshalled code object)
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if config is not None:
        code = apply_transforms(source, source_path, config, encrypt_fn)
    else:
        code = compile(source, source_path, 'exec', dont_inherit=True, optimize=2)

    magic = importlib.util.MAGIC_NUMBER
    flags = struct.pack('<I', 0)
    timestamp = struct.pack('<I', int(os.path.getmtime(source_path)))
    size = struct.pack('<I', os.path.getsize(source_path) & 0xFFFFFFFF)

    return magic + flags + timestamp + size + marshal.dumps(code)


def build_encrypted_package(source_dir: str, output_dir: str,
                            key: bytes, verbose: bool = True,
                            config: TransformConfig = None) -> dict:
    """Encrypt an entire Python package tree with optional transforms.

    For each .py file:
      1. Apply configured transforms (RFT, string encrypt, etc.)
      2. Compile to .pyc (bytecode, optimize=2)
      3. Encrypt .pyc with AES-256-GCM via Rust native
      4. Write as .enc to output_dir

    Args:
        source_dir: path to package root (must contain __init__.py)
        output_dir: output directory for .enc files
        key: 32-byte AES key
        verbose: print per-file progress
        config: TransformConfig for build-time transforms

    Returns:
        dict with {encrypted, failed, skipped, total_bytes, transforms}
    """
    from hevolvearmor._native import armor_encrypt

    # Create encrypt_fn closure for transforms that need it
    def _encrypt_fn(data):
        return bytes(armor_encrypt(data, key))

    stats = {
        'encrypted': 0, 'failed': 0, 'skipped': 0, 'total_bytes': 0,
        'transforms': {
            'rft_renamed': 0,
            'strings_encrypted': 0,
            'functions_wrapped': 0,
            'assert_imports_injected': 0,
            'private_mode_injected': 0,
        }
    }

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifest = []

    for dirpath, dirnames, filenames in os.walk(source_dir):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.endswith('.egg-info')]

        for fname in sorted(filenames):
            if not fname.endswith('.py'):
                continue

            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, source_dir).replace(os.sep, '/')

            enc_rel = rel_path.replace('.py', '.enc')
            enc_full = os.path.join(output_dir, enc_rel)
            os.makedirs(os.path.dirname(enc_full), exist_ok=True)

            try:
                pyc_bytes = compile_to_pyc_bytes(
                    full_path,
                    config=config,
                    encrypt_fn=_encrypt_fn if config else None,
                )
                encrypted = bytes(armor_encrypt(pyc_bytes, key))

                with open(enc_full, 'wb') as f:
                    f.write(encrypted)

                stats['encrypted'] += 1
                stats['total_bytes'] += len(encrypted)
                manifest.append(rel_path)

                if verbose:
                    flags = []
                    if config:
                        if config.rft_mode:
                            flags.append('RFT')
                        if config.encrypt_strings:
                            flags.append('STR')
                        if config.wrap_functions:
                            flags.append('WRAP')
                        if config.assert_imports:
                            flags.append('ASSERT')
                        if config.private_mode:
                            flags.append('PRIV')
                    flag_str = f" [{','.join(flags)}]" if flags else ""
                    print(f"  [OK] {rel_path} ({len(encrypted):,} bytes){flag_str}")

            except SyntaxError as e:
                stats['failed'] += 1
                if verbose:
                    print(f"  [FAIL] {rel_path}: SyntaxError: {e}")
            except Exception as e:
                stats['failed'] += 1
                if verbose:
                    print(f"  [FAIL] {rel_path}: {type(e).__name__}: {e}")

    # Write manifest
    manifest_path = os.path.join(output_dir, '_manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest))

    return stats
