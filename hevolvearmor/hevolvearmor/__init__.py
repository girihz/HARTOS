"""
HevolveArmor — Encrypted Python module loader for HART OS.

Runtime:
    import hevolvearmor
    hevolvearmor.install('/path/to/modules')
    import hevolveai  # transparently decrypted

Build-time:
    from hevolvearmor import encrypt_package
    encrypt_package('/path/to/hevolveai', '/path/to/output')

All security-critical logic (import hook, key derivation, anti-tamper,
anti-debug) lives in the compiled Rust binary (_native.pyd/.so).
"""
__version__ = "0.1.0"

# Re-export from Rust native — these are the ONLY public APIs
from hevolvearmor._native import (
    install,
    uninstall,
    derive_runtime_key,
    armor_encrypt,
    armor_decrypt,
    armor_generate_key,
    armor_derive_key_ed25519,
    armor_derive_key_passphrase,
    armor_derive_key_raw,
    armor_self_hash,
    armor_load_module,
    ArmoredFinder,
    ArmoredLoader,
    KEY_SIZE,
    NONCE_SIZE,
)

# Build-time helper (stays in Python — not security-critical)
from hevolvearmor._builder import build_encrypted_package as encrypt_package

__all__ = [
    "install",
    "uninstall",
    "derive_runtime_key",
    "encrypt_package",
    "armor_encrypt",
    "armor_decrypt",
    "armor_generate_key",
    "armor_self_hash",
    "KEY_SIZE",
]
