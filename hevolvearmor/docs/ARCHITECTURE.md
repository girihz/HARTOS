# HevolveArmor Architecture

> **Version**: 0.1.0
> **Status**: Encryption-at-rest IMPLEMENTED; advanced layers IMPLEMENTING
> **License**: Open Source (Rust core + Python shim)

---

## 1. Overview

HevolveArmor is a **Rust-native encrypted Python module loader** that protects Python
source code from reverse engineering. It encrypts `.py` files at build time into
`.enc` blobs using AES-256-GCM, then transparently decrypts them at import time via
a Rust-compiled `sys.meta_path` hook.

**Core crypto stack:**

| Primitive | Algorithm | Purpose |
|-----------|-----------|---------|
| Symmetric encryption | AES-256-GCM | Module encryption at rest |
| Key derivation | HKDF-SHA256 | Derive AES key from any input key material |
| Asymmetric identity | Ed25519 | Node key → deterministic AES key derivation |
| Integrity | SHA-256 | Binary self-hash (anti-tamper) |

**Why not PyArmor?**

- PyArmor's advanced protections (BCC, RFT) require paid licenses ($69-$512)
- PyArmor's runtime binary is proprietary and cannot be audited
- PyArmor does not integrate with Ed25519/HKDF key hierarchies
- HevolveArmor uses standard, auditable crypto primitives
- HevolveArmor's Rust source is fully open -- security by transparency, not obscurity

**Target audience:** Any Python package distributor. Default integration with the
HART OS key hierarchy, but fully configurable via `HEVOLVEARMOR_*` environment
variables for standalone use.

---

## 2. Architecture

### 2.1 Build-Time Pipeline

```
                       Python (_builder.py)              Rust (_native.pyd/.so)
                      ┌─────────────────────┐           ┌─────────────────────┐
  mypackage/          │                     │           │                     │
  ├── __init__.py ──> │ 1. Read .py source  │           │                     │
  ├── core.py ──────> │ 2. compile() → AST  │           │                     │
  ├── utils.py ─────> │ 3. marshal → .pyc   │ ── key ─> │ 4. AES-256-GCM     │
  └── ...             │    (optimize=2)      │ ── data > │    encrypt .pyc     │
                      └─────────────────────┘           │ 5. Write .enc       │
                                                        └─────────────────────┘
                                                                  │
                                                                  v
                                                        output/
                                                        ├── __init__.enc
                                                        ├── core.enc
                                                        ├── utils.enc
                                                        └── _manifest.txt
```

**Step-by-step:**

1. `_builder.py` walks the source tree, skipping `__pycache__`, `.git`, `tests`, etc.
2. Each `.py` file is compiled to a code object with `compile(..., optimize=2)` (strips
   docstrings and asserts).
3. The code object is marshalled to `.pyc` bytes with the standard 16-byte header:
   `magic(4) + flags(4) + timestamp(4) + source_size(4) + marshalled_code`.
4. The `.pyc` bytes are passed to Rust `armor_encrypt()` which generates a random
   12-byte nonce and encrypts with AES-256-GCM.
5. The encrypted blob is written as `.enc` with layout: `nonce(12) || ciphertext || tag(16)`.
6. A `_manifest.txt` listing all encrypted module paths is written alongside.

### 2.2 Runtime Pipeline

```
  import mypackage.core
       │
       v
  ┌─────────────────────────────────────────────────────────────┐
  │ Python import machinery (sys.meta_path)                     │
  │                                                             │
  │  1. ArmoredFinder.find_spec("mypackage.core")              │
  │     └── check top-level name ∈ package_names                │
  │     └── find_enc_path() → output/mypackage/core.enc         │
  │     └── return ModuleSpec(loader=ArmoredLoader)              │
  │                                                             │
  │  2. ArmoredLoader.exec_module(module)                       │
  │     ├── STATE.lock() → get key from ArmorState              │
  │     ├── fs::read("core.enc")                                │
  │     ├── AES-256-GCM decrypt (Rust) → .pyc bytes             │
  │     ├── Cache decrypted .pyc in code_cache                  │
  │     ├── marshal.loads(pyc[16:]) → code object               │
  │     └── exec(code, module.__dict__)                         │
  └─────────────────────────────────────────────────────────────┘
       │
       v
  mypackage.core is now a live Python module
  (code object in memory, .enc still encrypted on disk)
```

**All of the above runs inside the compiled Rust binary.** The Python-side
`ArmoredFinder`/`ArmoredLoader` in `_loader.py` exist only as a fallback; the
primary path is the Rust `#[pyclass]` implementation registered in `sys.meta_path`.

### 2.3 Key Derivation Chain

```
  Priority 1          Priority 2           Priority 3          Priority 4
  ┌──────────┐       ┌──────────────┐     ┌──────────────┐    ┌────────────┐
  │ Ed25519  │       │ HEVOLVE_     │     │ Tier-based   │    │ Passphrase │
  │ node key │       │ DATA_KEY     │     │ derivation   │    │ (manual)   │
  │ (.pem)   │       │ env var      │     │              │    │            │
  └────┬─────┘       └──────┬───────┘     └──────┬───────┘    └─────┬──────┘
       │                    │                    │                   │
       v                    v                    v                   v
  sign("hevolve      raw bytes              SHA256(               SHA256(
   armor-keygen         │                 "hevolvearmor-       passphrase +
   -v1")                │                  tier-{tier}-          salt)
       │                │                  {master_pk}")          │
       v                v                    │                    v
  sig[0:32]         ┌───────┐                v               ┌───────┐
       │            │ HKDF  │           ┌───────┐            │ HKDF  │
       v            │SHA256 │           │ HKDF  │            │SHA256 │
  ┌───────┐         └───┬───┘           │SHA256 │            └───┬───┘
  │ HKDF  │             │              └───┬───┘                 │
  │SHA256 │             v                  │                     v
  └───┬───┘        AES-256 key             v                AES-256 key
      │            (32 bytes)         AES-256 key            (32 bytes)
      v                               (32 bytes)
 AES-256 key
 (32 bytes)

  HKDF parameters (all paths):
    salt = b"hart-os-encrypted-modules-salt"
    info = b"hevolvearmor-v1-module-key"
```

**Key search locations (Ed25519):**

1. Explicit `node_key_path` argument
2. `$HEVOLVE_KEY_DIR/node_private_key.pem`
3. `./agent_data/node_private_key.pem`
4. `~/Documents/Nunba/data/node_private_key.pem`

All env var names are overridable for non-HART OS projects:

| Override Env Var | Default Value | Purpose |
|------------------|---------------|---------|
| `HEVOLVEARMOR_KEY_DIR_VAR` | `HEVOLVE_KEY_DIR` | Env var name for key directory |
| `HEVOLVEARMOR_DATA_KEY_VAR` | `HEVOLVE_DATA_KEY` | Env var name for data encryption key |
| `HEVOLVEARMOR_TIER_VAR` | `HEVOLVE_NODE_TIER` | Env var name for topology tier |
| `HEVOLVEARMOR_MASTER_PK` | *(HART OS default)* | Hex-encoded master public key |
| `HEVOLVEARMOR_APP_NAME` | `Nunba` | App name for Documents path |

---

## 3. Security Model

### 3.1 Threat Model

**Protecting against:** Reverse engineering of Python source code in distributed
deployments where end users have filesystem access to the installed package.

**Assumptions:**

- The attacker has full read access to the filesystem (can read `.enc` files)
- The attacker does NOT have the AES key or the Ed25519 private key
- The attacker may attempt to attach debuggers or patch the runtime binary
- The attacker may attempt to intercept decrypted bytecode in memory

**NOT protecting against:** An attacker with root access who can dump process memory
at arbitrary points. Memory-resident protections (per-function wrapping) raise the
bar but cannot prevent a sufficiently motivated kernel-level attacker.

### 3.2 Security Boundary: Rust vs Python

```
  ┌─────────────────────────────────────────────────────────┐
  │            RUST BINARY (_native.pyd / _native.so)       │
  │                                                         │
  │  Compiled, stripped, LTO'd — not trivially patchable    │
  │                                                         │
  │  ● Import hook (ArmoredFinder + ArmoredLoader)          │
  │  ● Key derivation (Ed25519/HKDF/passphrase)             │
  │  ● AES-256-GCM encrypt/decrypt                          │
  │  ● Key storage (zeroize-on-drop, Mutex-guarded)         │
  │  ● Anti-tamper (SHA-256 self-hash verification)         │
  │  ● Anti-debug (sys.settrace/sys.setprofile nulled)      │
  │  ● Code cache (decrypted .pyc bytes, in Rust heap)      │
  │  ● PEM parsing / base64 decoding                        │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │            PYTHON SHIM (thin, non-security-critical)    │
  │                                                         │
  │  ● __init__.py    — re-exports from _native             │
  │  ● _builder.py    — build-time only (compile .py→.pyc)  │
  │  ● _keygen.py     — Python-side key helper (optional)   │
  │  ● _loader.py     — fallback finder (mirrors Rust)      │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

The Python files are NOT part of the security boundary. They exist for:

- `_builder.py`: Build-time compilation (uses Python's `compile()` and `marshal`)
- `_loader.py`: Development/fallback import hook (mirrors the Rust implementation)
- `_keygen.py`: Optional Python-side key derivation (for environments where the
  Rust binary is not yet compiled)

In production, only the Rust binary handles key material and decryption.

### 3.3 Key Material Lifecycle

```
  derive_runtime_key_internal()
       │
       v
  key: [u8; 32]  ─────────> ArmorState.key (Mutex<Option<...>>)
       │                          │
       │                          │  on Drop: key.zeroize()
       │                          │  on uninstall(): key.zeroize(), state = None
       │                          │
       v                          v
  Returned to Python?         NEVER. Key stays in Rust heap.
  NO — derive_runtime_key()   Python only gets PyBytes copy
  zeroizes before return.     for build-time encrypt, then
                              the Python bytes object is GC'd.
```

Key material is **never exposed to the Python runtime** during normal operation.
The `derive_runtime_key()` function (exposed for build scripts) returns a copy
and immediately zeroizes the Rust-side array.

---

## 4. Protection Layers

### 4.1 Encryption at Rest (IMPLEMENTED)

**Status:** Fully implemented in Rust + Python builder.

**Algorithm:** AES-256-GCM (authenticated encryption with associated data).

**File format:**

```
  .enc file layout:
  ┌──────────┬─────────────────────────────────────┬──────────┐
  │ Nonce    │ Ciphertext                          │ GCM Tag  │
  │ 12 bytes │ len(plaintext) bytes                │ 16 bytes │
  └──────────┴─────────────────────────────────────┴──────────┘
  │<──────────────── total = 12 + len(.pyc) + 16 ───────────>│
```

- **Nonce**: 12 bytes, cryptographically random (`OsRng`), unique per file
- **Ciphertext**: The encrypted `.pyc` bytes (16-byte header + marshalled code)
- **GCM Tag**: 16-byte authentication tag (detects tampering of the ciphertext)

**Key derivation:** HKDF-SHA256 with domain-specific salt and info strings:

```rust
salt = b"hart-os-encrypted-modules-salt"
info = b"hevolvearmor-v1-module-key"
```

**Granularity:** Per-module. Each `.py` file produces one `.enc` file. This means
a corrupted or tampered `.enc` file only affects that single module; others remain
loadable.

**Performance:** ~50ms total for 143 modules at import time (measured on HART OS
`hevolveai` package). The overhead is dominated by filesystem reads, not decryption.

### 4.2 Per-Function Code Wrapping (IMPLEMENTING)

**Problem:** Once a module is decrypted, all function code objects are plaintext in
the Python interpreter's memory. A memory dump reveals all function bytecode.

**Solution:** Encrypt each function's `co_code` (bytecode) individually at build
time. At runtime, a Rust-injected wrapper decrypts the bytecode on call and
re-encrypts it on return.

```
  Build time:
  ┌────────────────────┐
  │ def process(data): │     AST rewrite        ┌──────────────────────┐
  │     result = ...   │ ──────────────────────> │ def process(data):   │
  │     return result  │                        │     _code = <blob>   │
  └────────────────────┘                        │     return _wrap(    │
                                                │       _code, data)   │
                                                └──────────────────────┘

  Runtime (Rust _wrap stub):
  1. Decrypt _code blob → original bytecode
  2. Create temporary code object
  3. Execute with provided arguments
  4. Zeroize decrypted bytecode
  5. Return result
```

**Performance:** ~0.1-0.5ms per function call. This is acceptable for most code
paths but problematic for hot loops.

**Selective wrapping:** Functions matching `--skip-wrap` patterns are encrypted
only at the module level (4.1), not individually wrapped. This is critical for
real-time inference paths.

```bash
# Wrap all _private functions, skip inference hot paths
python -m hevolvearmor encrypt ./pkg ./out \
    --wrap-functions "_*" \
    --skip-wrap "inference.*" --skip-wrap "forward_pass"
```

### 4.3 RFT Mode -- Symbol Renaming (IMPLEMENTING)

**Problem:** Even encrypted bytecode, once decrypted at runtime, contains readable
symbol names (`co_varnames`, `co_names`, `co_consts`). These reveal internal
architecture and logic.

**Solution:** AST-level renaming of internal symbols before compilation.

**Renaming rules:**

| Symbol Type | Renamed? | Rationale |
|-------------|----------|-----------|
| `_private_func()` | YES | Leading underscore = internal by Python convention |
| `_PrivateClass` | YES | Internal class |
| `__dunder__` methods | NO | Python runtime depends on exact names |
| `public_func()` | NO (default) | Part of public API |
| Local variables | YES | Never visible externally |
| `**kwargs` keys | NO | Callers depend on exact names |
| Serialized field names | NO | Would break pickle/JSON |
| `__init__` exports | NO | Public API contract |

**Risk:** Dynamic attribute access (`getattr(obj, '_some_name')`) breaks if the
target name is renamed. Mitigation: only rename `_private` prefixed symbols by
default, since the leading underscore is Python's convention for "internal."

**Modes:**

```bash
--rft-rename "_private_only"   # Only _prefixed symbols (safest)
--rft-rename "all_internal"    # All non-exported symbols
--rft-rename "aggressive"      # Everything except __dunder__ and __init__ exports
```

### 4.4 String Constant Encryption (IMPLEMENTING)

**Problem:** String literals (API URLs, prompt templates, config keys, error
messages) are visible in `co_consts` of decrypted bytecode.

**Solution:** At build time, replace string constants in `co_consts` with encrypted
blobs. At module load time, a Rust post-processing pass decrypts them before the
code object is executed.

```
  Build time:
  co_consts = ("https://api.example.com", "secret_key", 42)
                 │                           │
                 v                           v
  co_consts = (enc_blob_1,              enc_blob_2,         42)
              ^^^^^^^^^^                ^^^^^^^^^^
              AES-256-GCM              AES-256-GCM
              encrypted                encrypted

  Runtime (Rust, during exec_module):
  1. Walk co_consts
  2. Detect encrypted blobs (magic prefix)
  3. Decrypt in-place
  4. Return modified code object to exec()
```

**Performance:** ~1 microsecond per string decryption. Negligible even for modules with
hundreds of string constants.

**Config:**

```bash
--encrypt-strings                          # Encrypt all string constants
--encrypt-strings-filter "http*,api_*"     # Only matching patterns
```

### 4.5 BCC Mode -- Compile to C (IMPLEMENTING)

**Problem:** Python bytecode is reversible. Tools like `decompyle3` and `uncompyle6`
can reconstruct readable Python source from `.pyc` files.

**Solution:** Transpile Python functions to C code via Cython-like codegen, then
compile with the system C compiler. The resulting shared library is loaded instead
of bytecode. This is **irreversible** -- there is no bytecode to decompile.

```
  Build time:
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ module.py    │ ──> │ module.c     │ ──> │ module.so    │
  │ (Python)     │     │ (generated C │     │ (compiled    │
  │              │     │  via codegen) │     │  native)     │
  └──────────────┘     └──────────────┘     └──────────────┘
                       Uses CPython C API    Linked against
                       for all Python        libpython
                       object operations
```

**Limitations:**

- Only pure-Python modules (no inline C extensions)
- Skip modules with heavy `torch`/`numpy` interop (already compiled C under the hood)
- Requires system C compiler (`cc` / `cl.exe`) at build time
- Not needed at end-user runtime -- the `.so`/`.pyd` is pre-compiled

**Config:**

```bash
--bcc "mypackage.core.*"                   # BCC these modules
--skip-bcc "mypackage.models.*"            # Skip (torch interop)
```

### 4.6 License Management (IMPLEMENTING)

**Architecture:**

```
  ┌──────────────────────────────────────────────────┐
  │ License File (signed JWT)                        │
  │                                                  │
  │  Header: {"alg": "EdDSA", "typ": "JWT"}         │
  │  Payload: {                                      │
  │    "sub":     "customer-uuid",                   │
  │    "exp":     1735689600,                        │
  │    "iat":     1704153600,                        │
  │    "features": ["core", "inference", "training"],│
  │    "machine": {                                  │
  │      "mac":      "AA:BB:CC:DD:EE:FF",           │
  │      "hostname": "prod-node-1",                  │
  │      "disk":     "serial-12345"                  │
  │    },                                            │
  │    "grace_days": 7                               │
  │  }                                               │
  │  Signature: Ed25519(payload)                     │
  └──────────────────────────────────────────────────┘
```

**Machine binding:** Configurable combination of MAC address, hostname, and disk
serial number. The license is bound to a machine fingerprint hash, not individual
fields (so changing hostname alone does not invalidate).

**Runtime behavior:**

1. On `install()`, verify license signature and expiry
2. If expired but within grace period, log warning, continue
3. If expired beyond grace period, refuse to decrypt
4. Periodic re-validation every N seconds (configurable, default 3600)
5. Key rotation: new license file works with existing `.enc` files (the license
   controls access; the encryption key is derived from the key hierarchy, not the license)

**Config:**

```bash
python -m hevolvearmor license create \
    --expiry 2025-12-31 \
    --bind-machine \
    --features core,inference \
    --grace-days 7 \
    --period 3600
```

### 4.7 Anti-Debug (IMPLEMENTING)

**Purpose:** Detect and resist debugger attachment that could intercept decrypted
bytecode in memory.

**Platform-specific detection:**

| Platform | Method | API |
|----------|--------|-----|
| Linux | Self-attach ptrace | `ptrace(PTRACE_TRACEME, ...)` |
| Windows | Debugger presence | `IsDebuggerPresent()` + `NtQueryInformationProcess` |
| macOS | Kernel process info | `sysctl(CTL_KERN, KERN_PROC, KERN_PROC_PID, ...)` |
| All | Python trace hooks | `sys.settrace = None`, `sys.setprofile = None` |

**Current implementation (in Rust `install()`):**

```rust
// Null out Python-level debugging hooks
let _ = sys.setattr("settrace", py.None());
let _ = sys.setattr("setprofile", py.None());
```

**Enforcement:** Only active when `HEVOLVEARMOR_ENFORCE=1`. In development mode
(default), anti-debug is disabled so that normal debuggers and profilers work.

### 4.8 assert-import / assert-call (IMPLEMENTING)

**assert-import:** Before returning a decrypted module, verify that the module's
`.enc` file was loaded through the armored import hook (not monkey-patched or
replaced in `sys.modules`).

**assert-call:** At function entry, verify that the caller is itself an armored
module. This prevents unarmored code from calling protected functions.

```python
# Conceptual (implemented in Rust):
def protected_function():
    _assert_caller_armored()  # Rust check: inspect call stack
    ...                       # actual function body
```

**Configurable allowlist:** Trusted callers (e.g., the Python REPL, test frameworks)
can be allowlisted to bypass assert-call checks.

### 4.9 Private/Restrict Modes (IMPLEMENTING)

**Private mode:** Hide `_private` attributes from introspection.

```python
# Without private mode:
dir(mypackage)  # ['_internal_func', 'public_func', ...]

# With private mode:
dir(mypackage)  # ['public_func', ...]
getattr(mypackage, '_internal_func')  # AttributeError
```

Implemented via custom `__dir__()` and `__getattr__()` on the module object,
injected by the Rust loader during `exec_module`.

**Restrict mode:** Only armored code can import armored modules. Non-armored code
(e.g., user scripts) that attempts `import mypackage` gets an `ImportError`.

Implemented by inspecting the caller's `__spec__` in `find_spec` -- if the caller
is not itself an armored module, return `None` (module not found).

### 4.10 Cross-Platform Pre-built Runtimes (IMPLEMENTING)

**CI matrix:**

```
  GitHub Actions
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
  │  │  Linux    │  │  macOS    │  │  Windows   │   │
  │  │  x86_64   │  │  x86_64   │  │  x86_64    │   │
  │  │  aarch64  │  │  arm64    │  │            │   │
  │  └───────────┘  └───────────┘  └───────────┘   │
  │       ×              ×              ×            │
  │  Python 3.10    Python 3.10    Python 3.10      │
  │  Python 3.11    Python 3.11    Python 3.11      │
  │  Python 3.12    Python 3.12    Python 3.12      │
  │                                                 │
  │  = 15 wheel artifacts                           │
  │  Published to PyPI via maturin                  │
  └─────────────────────────────────────────────────┘
```

End users install with `pip install hevolvearmor` -- no Rust toolchain required.

---

## 5. Performance Considerations

### 5.1 Measured Overhead (Encryption at Rest, 143 modules)

| Operation | Time | Notes |
|-----------|------|-------|
| Full package encryption (build) | ~2.1s | Dominated by `compile()` + filesystem |
| Full package import (runtime) | ~50ms | AES-GCM decrypt + marshal + exec |
| Single module import | ~0.3ms | Cached after first decrypt |
| Key derivation | ~0.1ms | HKDF is fast; Ed25519 sign is ~0.2ms |

### 5.2 Projected Overhead (Advanced Layers)

| Layer | Per-Operation Cost | Impact |
|-------|-------------------|--------|
| Per-function wrapping (4.2) | 0.1-0.5ms/call | Significant for hot loops |
| String decryption (4.4) | ~1us/string | Negligible |
| RFT renaming (4.3) | 0ms (build-time only) | Zero runtime cost |
| BCC compiled C (4.5) | Negative (faster than bytecode) | Net positive |
| License check (4.6) | ~0.5ms/check | Only every N seconds |
| Anti-debug (4.7) | ~0.01ms at init | One-time |
| assert-import (4.8) | ~0.01ms/import | Negligible |

### 5.3 Recommended Configuration for Real-Time Inference

For HART OS `hevolveai` running inference at 100Hz:

```
  Full protection (all layers):
    ├── core/          — encryption + wrapping + RFT + strings
    ├── memory/        — encryption + wrapping + RFT + strings
    ├── config/        — encryption + wrapping + RFT + strings
    ├── tools/         — encryption + wrapping + RFT + strings
    └── validation/    — encryption + wrapping + RFT + strings

  Encryption-at-rest only (hot paths):
    ├── inference/     — encryption only (skip wrapping)
    ├── learning/      — encryption only (skip wrapping)
    └── models/        — encryption only (skip wrapping)
```

```bash
python -m hevolvearmor encrypt ./hevolveai ./output \
    --wrap-functions "_*" \
    --skip-wrap "inference.*" \
    --skip-wrap "learning.*" \
    --skip-wrap "models.*" \
    --rft-rename "_private_only" \
    --encrypt-strings
```

---

## 6. Comparison with PyArmor

| Feature | HevolveArmor | PyArmor (Free) | PyArmor (Pro $69) | PyArmor (Group $512) |
|---------|-------------|----------------|--------------------|-----------------------|
| **Encryption at rest** | AES-256-GCM | AES-256-CBC | AES-256-CBC | AES-256-CBC |
| **Authenticated encryption** | YES (GCM tag) | NO (CBC) | NO (CBC) | NO (CBC) |
| **Key derivation** | HKDF-SHA256 + Ed25519 | Proprietary | Proprietary | Proprietary |
| **Custom key hierarchy** | YES (configurable env vars) | NO | NO | NO |
| **Per-function wrapping** | IMPLEMENTING | NO | NO | NO |
| **RFT (symbol renaming)** | IMPLEMENTING | NO | YES | YES |
| **String encryption** | IMPLEMENTING | NO | NO | YES |
| **BCC (compile to C)** | IMPLEMENTING | NO | YES | YES |
| **License management** | IMPLEMENTING | Built-in | Built-in | Built-in |
| **Anti-debug** | IMPLEMENTING | YES | YES | YES |
| **assert-import/call** | IMPLEMENTING | NO | YES | YES |
| **Private/restrict modes** | IMPLEMENTING | NO | YES | YES |
| **Cross-platform wheels** | IMPLEMENTING | YES | YES | YES |
| **Runtime auditable** | YES (open Rust source) | NO (proprietary binary) | NO | NO |
| **Crypto auditable** | YES (standard primitives) | NO (custom implementation) | NO | NO |
| **Ed25519 integration** | Native | NO | NO | NO |
| **HART OS key hierarchy** | Native | NO | NO | NO |
| **Price** | Free (open source) | Free | $69 one-time | $512 one-time |
| **Python 3.10-3.12** | YES | YES | YES | YES |
| **Linux/macOS/Windows** | YES | YES | YES | YES |
| **ARM64 support** | YES | Limited | Limited | Limited |

---

## 7. Usage

### 7.1 Build-Time (Encrypt a Package)

**CLI:**

```bash
# Encrypt with passphrase
python -m hevolvearmor encrypt ./mypackage ./output --passphrase secret

# Encrypt with automatic HART OS key derivation
python -m hevolvearmor encrypt ./mypackage ./output

# Encrypt with explicit key file
python -m hevolvearmor encrypt ./mypackage ./output \
    --node-key ./path/to/node_private_key.pem
```

**Python API:**

```python
from hevolvearmor import encrypt_package, derive_runtime_key

# Derive key (uses HART OS hierarchy or falls back to passphrase)
key = derive_runtime_key(passphrase="my-build-secret")

# Encrypt
stats = encrypt_package(
    source_dir="./mypackage",
    output_dir="./output",
    key=key,
    verbose=True,
)
print(f"Encrypted {stats['encrypted']} modules ({stats['total_bytes']:,} bytes)")
```

**Output:**

```
  [OK] __init__.py (1,247 bytes)
  [OK] core.py (15,832 bytes)
  [OK] utils/helpers.py (3,291 bytes)
  [OK] utils/__init__.py (412 bytes)
  ...
  Encrypted 143 modules (1,284,519 bytes)
```

### 7.2 Runtime (Load Encrypted Modules)

**Minimal:**

```python
import hevolvearmor

# Install the import hook (key derived automatically)
hevolvearmor.install('/path/to/encrypted/modules', passphrase='secret')

# Now import as normal — decryption is transparent
import mypackage
mypackage.do_something()
```

**With explicit key path:**

```python
hevolvearmor.install(
    '/path/to/encrypted/modules',
    node_key_path='/path/to/node_private_key.pem',
)
```

**With anti-tamper verification:**

```python
hevolvearmor.install(
    '/path/to/encrypted/modules',
    passphrase='secret',
    expected_hash='a1b2c3d4...',  # SHA-256 of the _native.pyd/.so binary
)
```

**Cleanup:**

```python
hevolvearmor.uninstall()  # Removes finder, zeroizes key
```

### 7.3 HART OS Integration

In a HART OS deployment, key derivation is fully automatic:

```python
# The node already has an Ed25519 keypair (security/node_integrity.py)
# and HEVOLVE_DATA_KEY is set. No passphrase needed.
import hevolvearmor
hevolvearmor.install('/opt/hartos/armored_modules')

# Key derivation chain:
#   1. Try Ed25519 node key at $HEVOLVE_KEY_DIR/node_private_key.pem → FOUND
#   2. sign("hevolvearmor-keygen-v1") → HKDF → AES key
#   3. Done. Same key the build step used.
```

**Tier-based distribution:**

```bash
# Build encrypted modules for the "regional" tier
export HEVOLVE_NODE_TIER=regional
python -m hevolvearmor encrypt ./hevolveai ./dist/regional/

# Any "regional" node can decrypt (they all derive the same tier key)
# A "flat" node CANNOT decrypt (different tier → different key)
```

### 7.4 Standalone (Non-HART OS)

Override the env var names to use your own key infrastructure:

```bash
# Tell HevolveArmor to read YOUR env var instead of HEVOLVE_DATA_KEY
export HEVOLVEARMOR_DATA_KEY_VAR=MY_APP_SECRET_KEY
export MY_APP_SECRET_KEY=your-256-bit-key-here

# Encrypt
python -m hevolvearmor encrypt ./mypackage ./output

# Runtime (same env vars must be set)
python -c "
import hevolvearmor
hevolvearmor.install('./output')
import mypackage
"
```

Or use a simple passphrase (no env vars needed):

```bash
python -m hevolvearmor encrypt ./mypackage ./output --passphrase "my-secret"
python -c "
import hevolvearmor
hevolvearmor.install('./output', passphrase='my-secret')
import mypackage
"
```

---

## 8. Building from Source

### 8.1 Prerequisites

- **Rust** 1.70+ (`rustup` recommended)
- **Python** 3.10+
- **maturin** 1.0+ (`pip install maturin`)

### 8.2 Development Build

```bash
cd hevolvearmor

# Build Rust extension in-place (debug mode, fast compile)
maturin develop

# Build Rust extension in-place (release mode, optimized)
maturin develop --release

# Verify
python -c "from hevolvearmor._native import KEY_SIZE; print(f'OK: {KEY_SIZE}-byte keys')"
```

### 8.3 Run Tests

```bash
python -m pytest tests/ -v
```

### 8.4 Build Wheel for Distribution

```bash
# Build optimized wheel
maturin build --release

# Output: target/wheels/hevolvearmor-0.1.0-cp310-...-{platform}.whl
```

### 8.5 Release Profile

The `Cargo.toml` release profile is tuned for security:

```toml
[profile.release]
opt-level = 3        # Maximum optimization
lto = true           # Link-Time Optimization (smaller, harder to RE)
strip = true         # Strip debug symbols
codegen-units = 1    # Single codegen unit (better optimization, harder to RE)
panic = "abort"      # No unwinding (smaller binary, no stack traces)
```

---

## 9. Security Considerations

### 9.1 The Rust Binary IS the Security Boundary

If an attacker can modify the `_native.pyd` / `_native.so` binary, all protections
fall. The anti-tamper self-hash (Section 4.7) detects this, but only if the expected
hash is provided at `install()` time.

**Mitigation chain:**

1. **strip + LTO + codegen-units=1**: Makes the binary harder to reverse engineer
2. **Anti-tamper self-hash**: Detects binary modification at runtime
3. **Code signing** (OS-level): Sign the `.pyd`/`.so` with a code signing certificate
4. **Deployment**: Ship via signed packages (PyPI wheels are hash-verified)

### 9.2 Key Material Handling

- AES keys are stored in a `[u8; 32]` array that implements `Zeroize`
- On `Drop` (process exit, `uninstall()`, or state replacement), the key is overwritten
  with zeros
- The key is **never** exposed to Python at runtime -- it lives inside the Rust
  `Mutex<Option<ArmorState>>` and is only accessed by Rust decryption functions
- The `derive_runtime_key()` Python function returns a copy and immediately zeroizes
  the Rust-side array; this function is intended for build scripts only

### 9.3 Thread Safety

All mutable global state is behind `static STATE: Mutex<Option<ArmorState>>`.
Multiple threads can import armored modules concurrently; the mutex ensures that
key material and the code cache are accessed safely.

### 9.4 Cryptographic Choices

| Choice | Rationale |
|--------|-----------|
| AES-256-GCM over AES-CBC | GCM provides authentication (tamper detection); CBC does not |
| HKDF-SHA256 over raw hashing | HKDF is the standard extract-then-expand KDF (RFC 5869) |
| Ed25519 over RSA | Smaller keys (32 bytes), faster signing, deterministic |
| 12-byte random nonce | Standard for GCM; collision probability negligible at module-level granularity |
| Per-file nonces (not counter) | Avoids nonce-reuse bugs from interrupted builds or parallel encryption |

### 9.5 What This Does NOT Protect Against

- **Root-level memory dumps**: A root user can dump process memory and find decrypted
  code objects. Per-function wrapping (4.2) reduces the window but cannot eliminate it.
- **Python-level monkey-patching before install()**: If attacker code runs before
  `hevolvearmor.install()`, it can intercept imports. Mitigate by calling `install()`
  as early as possible (ideally in `sitecustomize.py`).
- **Recompiled Rust binary**: An attacker who builds a modified `_native.so` that
  dumps keys can bypass everything. This is inherent to any client-side protection.

---

## 10. Rationale: Why Not Just Use PyArmor?

### 10.1 Licensing

PyArmor's free tier only provides basic encryption (AES-CBC without authentication).
Advanced protections require paid licenses:

- **RFT mode** (symbol renaming): Pro license ($69)
- **BCC mode** (compile to C): Pro license ($69)
- **String encryption**: Group license ($512)
- **assert-import/call**: Pro license ($69)

HevolveArmor provides all of these as open-source, free for any use.

### 10.2 Auditability

PyArmor's runtime (`pytransform3`) is a proprietary compiled binary. Users must trust
that:

- The crypto implementation is correct
- There are no backdoors
- Key material is properly handled

HevolveArmor's Rust source is fully open. Anyone can audit the AES-256-GCM
implementation (which delegates to the `aes-gcm` crate, itself widely audited),
the HKDF key derivation, and the key zeroization logic.

### 10.3 Key Hierarchy Integration

PyArmor uses its own key management. It cannot integrate with:

- Ed25519 node identities
- HKDF-based key derivation chains
- Tier-based key distribution
- Existing `HEVOLVE_DATA_KEY` infrastructure

HevolveArmor was designed from the ground up to plug into the HART OS security
model while remaining fully configurable for other projects.

### 10.4 Authenticated Encryption

PyArmor uses AES-CBC, which provides confidentiality but not integrity. A corrupted
or tampered `.enc` file may decrypt to garbage without raising an error. AES-256-GCM
(used by HevolveArmor) includes a 16-byte authentication tag that detects any
modification to the ciphertext.

---

## Appendix A: File Layout

```
hevolvearmor/
├── Cargo.toml                 # Rust crate manifest
├── Cargo.lock                 # Rust dependency lock
├── pyproject.toml             # Python package manifest (maturin backend)
├── src/
│   └── lib.rs                 # ALL Rust code (security boundary)
├── hevolvearmor/
│   ├── __init__.py            # Re-exports from _native
│   ├── _builder.py            # Build-time: .py → .pyc → .enc
│   ├── _keygen.py             # Python-side key derivation (optional)
│   └── _loader.py             # Fallback import hook (mirrors Rust)
├── docs/
│   └── ARCHITECTURE.md        # This document
└── target/                    # Rust build artifacts (gitignored)
```

## Appendix B: Rust Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `pyo3` | 0.23 | Python ↔ Rust FFI |
| `aes-gcm` | 0.10 | AES-256-GCM encryption |
| `sha2` | 0.10 | SHA-256 hashing |
| `hkdf` | 0.12 | HKDF key derivation (RFC 5869) |
| `ed25519-dalek` | 2.x | Ed25519 signing for key derivation |
| `rand` | 0.8 | Cryptographic RNG (nonce generation) |
| `hex` | 0.4 | Hex encoding (hash display) |
| `serde` / `serde_json` | 1.x | JSON serialization (manifest, config) |
| `zeroize` | 1.x | Secure memory wiping |
| `dirs-next` | 2.x | Platform documents directory |

## Appendix C: Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `HEVOLVEARMOR_KEY_DIR_VAR` | `HEVOLVE_KEY_DIR` | Name of env var containing key directory path |
| `HEVOLVEARMOR_DATA_KEY_VAR` | `HEVOLVE_DATA_KEY` | Name of env var containing data encryption key |
| `HEVOLVEARMOR_TIER_VAR` | `HEVOLVE_NODE_TIER` | Name of env var containing topology tier |
| `HEVOLVEARMOR_MASTER_PK` | *(HART OS hex)* | Hex-encoded Ed25519 master public key |
| `HEVOLVEARMOR_APP_NAME` | `Nunba` | Application name (used for Documents path lookup) |
| `HEVOLVEARMOR_ENFORCE` | `0` | Enable anti-debug enforcement (`1` = armed) |
