"""
CLI for HevolveArmor — encrypt Python packages with AES-256-GCM.

Usage:
    python -m hevolvearmor encrypt <source> <output> [--passphrase X] [--key-file F]
    python -m hevolvearmor keygen [--passphrase X] [--output F]
    python -m hevolvearmor verify <modules_dir> [--passphrase X]
    python -m hevolvearmor hash    # print self-hash of the native binary

Examples:
    # Encrypt a package
    python -m hevolvearmor encrypt ./mypackage ./encrypted --passphrase secret

    # Generate a key file
    python -m hevolvearmor keygen --output my.key

    # Verify encrypted modules can be decrypted
    python -m hevolvearmor verify ./encrypted --passphrase secret
"""
import argparse
import os
import sys


def cmd_encrypt(args):
    from hevolvearmor import encrypt_package, derive_runtime_key, armor_generate_key

    if args.key_file and os.path.isfile(args.key_file):
        with open(args.key_file, 'rb') as f:
            key = f.read()
        if len(key) != 32:
            print(f"Error: key file must be 32 bytes, got {len(key)}")
            sys.exit(1)
    elif args.passphrase:
        key = derive_runtime_key(passphrase=args.passphrase)
    else:
        key = armor_generate_key()
        # Save generated key
        key_path = args.key_file or os.path.join(args.output, '..', '_key.bin')
        key_path = os.path.abspath(key_path)
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'wb') as f:
            f.write(key)
        print(f"Generated key saved to: {key_path}")

    stats = encrypt_package(args.source, args.output, key, verbose=not args.quiet)

    if not args.quiet:
        print(f"\nEncrypted: {stats['encrypted']}, Failed: {stats['failed']}, "
              f"Size: {stats['total_bytes'] / 1024:.0f} KB")

    if stats['failed'] > 0:
        sys.exit(1)


def cmd_keygen(args):
    from hevolvearmor import derive_runtime_key, armor_generate_key

    if args.passphrase:
        key = derive_runtime_key(passphrase=args.passphrase)
        print(f"Derived key from passphrase ({len(key)} bytes)")
    elif args.ed25519:
        key = derive_runtime_key(node_key_path=args.ed25519)
        print(f"Derived key from Ed25519 key ({len(key)} bytes)")
    else:
        key = armor_generate_key()
        print(f"Generated random key ({len(key)} bytes)")

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(key)
        print(f"Saved to: {args.output}")
    else:
        print(f"Hex: {key.hex()}")


def cmd_verify(args):
    from hevolvearmor import install, uninstall

    try:
        install(
            args.modules_dir,
            passphrase=args.passphrase,
            node_key_path=args.key_file,
        )
        print("Loader installed successfully")

        # Try importing each top-level package
        import importlib
        for entry in os.listdir(args.modules_dir):
            entry_path = os.path.join(args.modules_dir, entry)
            if os.path.isdir(entry_path) and os.path.isfile(
                    os.path.join(entry_path, '__init__.enc')):
                try:
                    mod = importlib.import_module(entry)
                    print(f"  [OK] {entry}")
                except Exception as e:
                    print(f"  [FAIL] {entry}: {e}")

        uninstall()
        print("Verification complete")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_bcc(args):
    from hevolvearmor._bcc import compile_package_to_c

    patterns = args.patterns.split(',') if args.patterns else None
    skip = args.skip.split(',') if args.skip else None

    stats = compile_package_to_c(
        args.source, args.output,
        patterns=patterns, skip_patterns=skip,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"\nCompiled: {stats['compiled']}, Failed: {stats['failed']}, "
              f"Skipped: {stats['skipped']}")


def cmd_hash(args):
    from hevolvearmor import armor_self_hash
    h = armor_self_hash()
    print(h)


def main():
    parser = argparse.ArgumentParser(
        prog='hevolvearmor',
        description='HevolveArmor — Encrypt Python packages with AES-256-GCM + Ed25519 key derivation',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # encrypt
    p_enc = sub.add_parser('encrypt', help='Encrypt a Python package')
    p_enc.add_argument('source', help='Path to Python package root')
    p_enc.add_argument('output', help='Output directory for .enc files')
    p_enc.add_argument('--passphrase', '-p', help='Derive key from passphrase')
    p_enc.add_argument('--key-file', '-k', help='Path to 32-byte key file')
    p_enc.add_argument('--quiet', '-q', action='store_true')

    # keygen
    p_key = sub.add_parser('keygen', help='Generate or derive a key')
    p_key.add_argument('--passphrase', '-p', help='Derive from passphrase')
    p_key.add_argument('--ed25519', help='Derive from Ed25519 PEM file')
    p_key.add_argument('--output', '-o', help='Save key to file')

    # verify
    p_ver = sub.add_parser('verify', help='Verify encrypted modules')
    p_ver.add_argument('modules_dir', help='Path to encrypted modules')
    p_ver.add_argument('--passphrase', '-p')
    p_ver.add_argument('--key-file', '-k')

    # bcc
    p_bcc = sub.add_parser('bcc', help='Compile Python to C extensions (BCC mode)')
    p_bcc.add_argument('source', help='Path to Python package root')
    p_bcc.add_argument('output', help='Output directory for compiled extensions')
    p_bcc.add_argument('--patterns', help='Comma-separated fnmatch patterns to compile')
    p_bcc.add_argument('--skip', help='Comma-separated fnmatch patterns to skip')
    p_bcc.add_argument('--quiet', '-q', action='store_true')

    # hash
    sub.add_parser('hash', help='Print SHA-256 hash of native binary')

    args = parser.parse_args()
    {'encrypt': cmd_encrypt, 'keygen': cmd_keygen,
     'verify': cmd_verify, 'bcc': cmd_bcc, 'hash': cmd_hash}[args.command](args)


if __name__ == '__main__':
    main()
