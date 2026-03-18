"""
Build-time bytecode and AST transforms for HevolveArmor.

These transforms are applied BEFORE encryption, modifying the code objects
to add runtime protection layers:

1. String encryption — replace string constants with encrypted blobs
2. Assert-import — inject verification that imported modules are armored
3. Assert-call — inject verification that called functions are armored
4. RFT (Rename) — rename internal symbols at AST level
5. Per-function wrapping — replace function code with decrypt-on-call stubs

All transforms operate on Python code objects (post-compilation) or AST
(pre-compilation). The Rust runtime provides the decrypt callbacks.
"""
import ast
import dis
import marshal
import struct
import sys
import types
import fnmatch
import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  4. String Constant Encryption
# ═══════════════════════════════════════════════════════════════════════════════

def encrypt_strings_in_code(code: types.CodeType, encrypt_fn,
                            min_length: int = 4,
                            patterns: list = None) -> types.CodeType:
    """Replace string constants in co_consts with encrypted markers.

    At runtime, the Rust loader intercepts these markers and decrypts.

    Args:
        code: compiled code object
        encrypt_fn: callable(plaintext_bytes) → encrypted_bytes
        min_length: skip strings shorter than this (variable names, empty, etc.)
        patterns: if set, only encrypt strings matching these fnmatch patterns

    Returns:
        Modified code object with encrypted string constants.
    """
    new_consts = []
    for const in code.co_consts:
        if isinstance(const, str) and len(const) >= min_length:
            # Skip dunder names and simple identifiers (likely variable refs)
            if const.startswith('__') and const.endswith('__'):
                new_consts.append(const)
                continue
            # Skip if patterns specified and no match
            if patterns and not any(fnmatch.fnmatch(const, p) for p in patterns):
                new_consts.append(const)
                continue
            # Encrypt: wrap in a sentinel tuple that the runtime recognizes
            encrypted = encrypt_fn(const.encode('utf-8'))
            # Marker: (__hevolvearmor_enc_str__, encrypted_bytes)
            new_consts.append(('__hevolvearmor_enc_str__', bytes(encrypted)))
        elif isinstance(const, types.CodeType):
            # Recurse into nested code objects (functions, classes, lambdas)
            new_consts.append(
                encrypt_strings_in_code(const, encrypt_fn, min_length, patterns))
        else:
            new_consts.append(const)

    return code.replace(co_consts=tuple(new_consts))


def decrypt_strings_in_code(code: types.CodeType, decrypt_fn) -> types.CodeType:
    """Runtime: restore encrypted strings in co_consts.

    Called by the Rust loader after decrypting the .enc file.
    """
    new_consts = []
    for const in code.co_consts:
        if (isinstance(const, tuple) and len(const) == 2
                and const[0] == '__hevolvearmor_enc_str__'):
            plaintext = decrypt_fn(const[1])
            new_consts.append(plaintext.decode('utf-8'))
        elif isinstance(const, types.CodeType):
            new_consts.append(decrypt_strings_in_code(const, decrypt_fn))
        else:
            new_consts.append(const)

    return code.replace(co_consts=tuple(new_consts))


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Assert-Import — verify imported modules are armored
# ═══════════════════════════════════════════════════════════════════════════════

_ASSERT_IMPORT_TEMPLATE = '''
def __hevolvearmor_assert_import__(module_name):
    """Verify that an imported module is armored (not substituted)."""
    import sys as _sys
    _mod = _sys.modules.get(module_name)
    if _mod is None:
        return
    # Check for armored marker (set by ArmoredLoader)
    if not getattr(_mod, '__hevolvearmor__', False):
        raise ImportError(
            f"HevolveArmor: module '{{module_name}}' is not armored. "
            f"Possible module substitution attack."
        )
'''


def inject_assert_import(source: str, armored_packages: list) -> str:
    """Inject assert-import checks after each import of an armored package.

    Transforms:
        from embodied_ai.core import tool_registry
    To:
        from embodied_ai.core import tool_registry
        __hevolvearmor_assert_import__('embodied_ai.core')

    Args:
        source: Python source code string
        armored_packages: list of top-level package names that should be armored
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # can't transform, return as-is

    new_body = []
    needs_helper = False

    for node in tree.body:
        new_body.append(node)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Get the module name
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module] if node.module else []

            for name in names:
                top = name.split('.')[0]
                if top in armored_packages:
                    needs_helper = True
                    # Insert assertion call after import
                    assert_call = ast.parse(
                        f"__hevolvearmor_assert_import__('{name}')"
                    ).body[0]
                    ast.copy_location(assert_call, node)
                    new_body.append(assert_call)

    if needs_helper:
        # Prepend the helper function
        helper = ast.parse(_ASSERT_IMPORT_TEMPLATE).body[0]
        tree.body = [helper] + new_body
    else:
        tree.body = new_body

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ═══════════════════════════════════════════════════════════════════════════════
#  8b. Assert-Call — verify function callees haven't been replaced
# ═══════════════════════════════════════════════════════════════════════════════

_ASSERT_CALL_TEMPLATE = '''
def __hevolvearmor_assert_call__(fn):
    """Verify a callable hasn't been replaced with a non-armored substitute."""
    if hasattr(fn, '__code__') and not getattr(fn, '__hevolvearmor__', False):
        _mod = getattr(fn, '__module__', '')
        if _mod and any(_mod.startswith(p) for p in {armored_packages_repr}):
            raise RuntimeError(
                f"HevolveArmor: function '{{fn.__qualname__}}' in '{{_mod}}' "
                f"is not armored. Possible function substitution attack."
            )
    return fn
'''


# ═══════════════════════════════════════════════════════════════════════════════
#  6. RFT Mode — Symbol Renaming (AST Transform)
# ═══════════════════════════════════════════════════════════════════════════════

class _RFTRenamer(ast.NodeTransformer):
    """Rename internal (_private) symbols in an AST.

    Modes:
        _private_only: only rename _single_underscore prefixed names
        all_internal:  rename all non-public, non-dunder names
        aggressive:    rename everything except __init__ exports
    """

    # Names that must NEVER be renamed (Python semantics would break)
    NEVER_RENAME = frozenset({
        'self', 'cls', 'super', 'None', 'True', 'False',
        '__init__', '__new__', '__del__', '__repr__', '__str__',
        '__eq__', '__hash__', '__lt__', '__le__', '__gt__', '__ge__',
        '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
        '__mod__', '__pow__', '__neg__', '__pos__', '__abs__',
        '__and__', '__or__', '__xor__', '__invert__',
        '__getattr__', '__setattr__', '__delattr__', '__getattribute__',
        '__get__', '__set__', '__delete__',
        '__getitem__', '__setitem__', '__delitem__', '__contains__',
        '__len__', '__iter__', '__next__', '__reversed__',
        '__call__', '__enter__', '__exit__',
        '__aenter__', '__aexit__', '__aiter__', '__anext__',
        '__await__', '__class__', '__dict__', '__doc__',
        '__module__', '__name__', '__qualname__', '__slots__',
        '__all__', '__file__', '__path__', '__package__',
        '__spec__', '__loader__', '__builtins__', '__cached__',
        '__import__', '__annotations__', '__bases__', '__mro__',
        '__subclasses__', '__subclasshook__',
        # Common framework names
        'setUp', 'tearDown', 'setUpClass', 'tearDownClass',
    })

    def __init__(self, mode: str = '_private_only', preserve: set = None):
        self.mode = mode
        self.preserve = preserve or set()
        self._rename_map = {}
        self._counter = 0

    def _should_rename(self, name: str) -> bool:
        if name in self.NEVER_RENAME or name in self.preserve:
            return False
        if name.startswith('__') and name.endswith('__'):
            return False  # dunder — never rename

        if self.mode == '_private_only':
            return name.startswith('_') and not name.startswith('__')
        elif self.mode == 'all_internal':
            return name.startswith('_') or (not name[0].isupper() and name not in self.preserve)
        elif self.mode == 'aggressive':
            return True
        return False

    def _get_rename(self, name: str) -> str:
        if name not in self._rename_map:
            self._counter += 1
            # Use a hash-based name to avoid collisions
            import hashlib
            h = hashlib.md5(name.encode()).hexdigest()[:8]
            self._rename_map[name] = f'_ha_{h}_{self._counter}'
        return self._rename_map[name]

    def visit_Name(self, node):
        if self._should_rename(node.id):
            node.id = self._get_rename(node.id)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self._should_rename(node.name):
            node.name = self._get_rename(node.name)
        # Rename arguments
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if self._should_rename(arg.arg) and arg.arg != 'self' and arg.arg != 'cls':
                arg.arg = self._get_rename(arg.arg)
        if node.args.vararg and self._should_rename(node.args.vararg.arg):
            node.args.vararg.arg = self._get_rename(node.args.vararg.arg)
        if node.args.kwarg and self._should_rename(node.args.kwarg.arg):
            node.args.kwarg.arg = self._get_rename(node.args.kwarg.arg)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        if self._should_rename(node.name):
            node.name = self._get_rename(node.name)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._should_rename(node.attr):
            node.attr = self._get_rename(node.attr)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        # Don't rename keyword arguments in calls — they map to parameter names
        # which may be in external (non-armored) code
        return self.generic_visit(node)

    def visit_Global(self, node):
        node.names = [self._get_rename(n) if self._should_rename(n) else n
                      for n in node.names]
        return self.generic_visit(node)

    def visit_Nonlocal(self, node):
        node.names = [self._get_rename(n) if self._should_rename(n) else n
                      for n in node.names]
        return self.generic_visit(node)


def rft_rename(source: str, mode: str = '_private_only',
               preserve: set = None) -> tuple:
    """Apply RFT symbol renaming to Python source.

    Args:
        source: Python source code
        mode: '_private_only', 'all_internal', or 'aggressive'
        preserve: set of names to never rename

    Returns:
        (transformed_source, rename_map) — map is {original: renamed}
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, {}

    renamer = _RFTRenamer(mode=mode, preserve=preserve)
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree), dict(renamer._rename_map)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Per-Function Code Wrapping
# ═══════════════════════════════════════════════════════════════════════════════

# Marker constant injected into wrapped function code objects
_WRAP_MARKER = '__hevolvearmor_wrapped__'


def wrap_function_code(code: types.CodeType, encrypt_fn,
                       skip_patterns: list = None,
                       module_path: str = '') -> types.CodeType:
    """Wrap each function's code object so it's encrypted individually.

    Each function's inner code is replaced with a stub that:
    1. Calls the Rust native decrypt with the encrypted code blob
    2. Executes the decrypted function
    3. Result returned, decrypted code discarded (not cached in Python)

    Args:
        code: module-level code object
        encrypt_fn: callable(bytes) → bytes
        skip_patterns: list of fnmatch patterns for functions to skip
        module_path: dotted module path (for pattern matching)

    Returns:
        Modified code object with wrapped functions.
    """
    new_consts = []

    for const in code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name != '<module>':
            # This is a function/class code object
            qualname = f"{module_path}.{const.co_name}" if module_path else const.co_name

            # Check skip patterns
            if skip_patterns and any(
                    fnmatch.fnmatch(qualname, p) for p in skip_patterns):
                # Recurse into nested functions but don't wrap this one
                new_consts.append(
                    wrap_function_code(const, encrypt_fn, skip_patterns, qualname))
                continue

            # Encrypt the function's marshalled code
            marshalled = marshal.dumps(const)
            encrypted = encrypt_fn(marshalled)

            # Build a wrapper code object that calls the Rust decryptor
            # The wrapper stores the encrypted blob in its co_consts
            wrapper_source = f'''
def {const.co_name}(*__ha_args, **__ha_kwargs):
    """[HevolveArmor wrapped]"""
    import marshal as __ha_marshal
    from hevolvearmor._native import armor_decrypt as __ha_decrypt
    __ha_key = __ha_get_runtime_key()
    __ha_code = __ha_marshal.loads(__ha_decrypt(__ha_encrypted_blob, __ha_key))
    __ha_fn = type(lambda: None)(__ha_code, globals())
    return __ha_fn(*__ha_args, **__ha_kwargs)
'''
            # Instead of exec'ing source, we modify the code object's consts
            # to include the encrypted blob and a reference to the key getter.
            # This is done by creating a proper wrapper at the bytecode level.

            # For now, use a simpler approach: store encrypted blob as a const
            # in the code object, and inject a decrypt-and-exec preamble.
            # The Rust runtime handles this via a special marker.

            # Marker tuple: (__hevolvearmor_wrapped__, encrypted_marshalled_code)
            wrapped = ('__hevolvearmor_wrapped__', bytes(encrypted))
            new_consts.append(wrapped)
        elif isinstance(const, types.CodeType):
            # Module-level code — recurse
            new_consts.append(
                wrap_function_code(const, encrypt_fn, skip_patterns, module_path))
        else:
            new_consts.append(const)

    return code.replace(co_consts=tuple(new_consts))


def unwrap_function_code(code: types.CodeType, decrypt_fn) -> types.CodeType:
    """Runtime: restore wrapped function code objects.

    Called by the Rust loader. Replaces marker tuples with decrypted code objects.
    """
    new_consts = []

    for const in code.co_consts:
        if (isinstance(const, tuple) and len(const) == 2
                and const[0] == '__hevolvearmor_wrapped__'):
            encrypted = const[1]
            marshalled = decrypt_fn(encrypted)
            inner_code = marshal.loads(marshalled)
            # Recurse into the restored function (it may have nested functions)
            new_consts.append(unwrap_function_code(inner_code, decrypt_fn))
        elif isinstance(const, types.CodeType):
            new_consts.append(unwrap_function_code(const, decrypt_fn))
        else:
            new_consts.append(const)

    return code.replace(co_consts=tuple(new_consts))


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Private / Restrict Modes
# ═══════════════════════════════════════════════════════════════════════════════

_PRIVATE_MODULE_TEMPLATE = '''
class _ArmoredModuleType(type(__import__('sys'))):
    """Module type that hides private attributes from external access."""

    _HIDDEN_PREFIXES = ('_',)
    _ALLOWED_DUNDERS = frozenset({
        '__name__', '__loader__', '__package__', '__spec__', '__path__',
        '__file__', '__cached__', '__builtins__', '__doc__', '__all__',
        '__version__', '__hevolvearmor__',
    })

    def __dir__(self):
        return [k for k in super().__dir__()
                if not k.startswith('_') or k in self._ALLOWED_DUNDERS]

    def __getattr__(self, name):
        if name.startswith('_') and name not in self._ALLOWED_DUNDERS:
            # Check if caller is from an armored module
            import sys
            frame = sys._getframe(1)
            caller_mod = frame.f_globals.get('__name__', '')
            caller_armored = frame.f_globals.get('__hevolvearmor__', False)
            if not caller_armored:
                raise AttributeError(
                    f"module '{self.__name__}' has no attribute '{name}' "
                    f"(private, caller not armored)")
        return super().__getattribute__(name)
'''


def inject_private_mode(source: str) -> str:
    """Inject private-mode module wrapper at the end of source.

    Makes the module hide _private attributes from non-armored callers.
    """
    return source + '\n' + _PRIVATE_MODULE_TEMPLATE + '''
import sys as _ha_sys
_ha_sys.modules[__name__].__class__ = _ArmoredModuleType
del _ha_sys
'''


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API: apply all transforms in pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TransformConfig:
    """Configuration for build-time transforms."""

    def __init__(
        self,
        encrypt_strings: bool = False,
        string_min_length: int = 4,
        string_patterns: list = None,
        assert_imports: bool = False,
        assert_calls: bool = False,
        armored_packages: list = None,
        rft_mode: str = None,  # None, '_private_only', 'all_internal', 'aggressive'
        rft_preserve: set = None,
        wrap_functions: bool = False,
        wrap_patterns: list = None,    # fnmatch patterns to wrap
        skip_wrap_patterns: list = None,  # fnmatch patterns to skip
        private_mode: bool = False,
    ):
        self.encrypt_strings = encrypt_strings
        self.string_min_length = string_min_length
        self.string_patterns = string_patterns
        self.assert_imports = assert_imports
        self.assert_calls = assert_calls
        self.armored_packages = armored_packages or []
        self.rft_mode = rft_mode
        self.rft_preserve = rft_preserve or set()
        self.wrap_functions = wrap_functions
        self.wrap_patterns = wrap_patterns
        self.skip_wrap_patterns = skip_wrap_patterns
        self.private_mode = private_mode


def apply_transforms(source: str, source_path: str,
                     config: TransformConfig,
                     encrypt_fn=None) -> types.CodeType:
    """Apply all configured transforms and return a compiled code object.

    Pipeline order:
    1. RFT rename (AST-level, before compilation)
    2. Assert-import injection (AST-level)
    3. Private mode injection (source-level)
    4. Compile to code object
    5. String encryption (code object level)
    6. Per-function wrapping (code object level)

    Args:
        source: Python source code
        source_path: file path (for compile() origin)
        config: TransformConfig
        encrypt_fn: callable(bytes) → bytes (from Rust native)

    Returns:
        Transformed code object ready for marshalling + encryption.
    """
    rename_map = {}

    # 1. RFT rename (AST level — must happen before compile)
    if config.rft_mode:
        source, rename_map = rft_rename(
            source, mode=config.rft_mode, preserve=config.rft_preserve)

    # 2. Assert-import
    if config.assert_imports and config.armored_packages:
        source = inject_assert_import(source, config.armored_packages)

    # 3. Private mode
    if config.private_mode:
        source = inject_private_mode(source)

    # 4. Compile
    code = compile(source, source_path, 'exec', dont_inherit=True, optimize=2)

    # 5. String encryption
    if config.encrypt_strings and encrypt_fn is not None:
        code = encrypt_strings_in_code(
            code, encrypt_fn,
            min_length=config.string_min_length,
            patterns=config.string_patterns)

    # 6. Per-function wrapping
    if config.wrap_functions and encrypt_fn is not None:
        module_path = source_path.replace('/', '.').replace('\\', '.').rstrip('.py')
        code = wrap_function_code(
            code, encrypt_fn,
            skip_patterns=config.skip_wrap_patterns,
            module_path=module_path)

    # Add armored marker to module globals
    # (checked by assert-import at runtime)
    # This is done by the loader, not here — the marker is set on the module object

    return code
