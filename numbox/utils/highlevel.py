import hashlib
import os
import re
import tempfile
from inspect import getmodule, getsource
from io import StringIO
from pathlib import Path
from numba import njit
from numba.core.itanium_mangler import mangle_type_or_value
from numba.core.types import Type, intp, void
from numba.core.types.functions import Dispatcher
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.experimental.function_type import FunctionType
from numba.experimental.structref import define_boxing, new, StructRefProxy
from numba.extending import overload, overload_method
from textwrap import dedent, indent
from typing import Callable, Iterable, Optional

from numbox.core.configurations import default_jit_options
from numbox.utils.standard import make_params_strings
from numbox.utils._addr_global import (
    _load_addr_from_named_global,
    _make_icall_for_sig,
    _store_addr_to_named_global,
)


def _file_anchor():
    raise NotImplementedError


def cres(sig, **kwargs):
    """ Returns Python proxy to `FunctionType` rather than `CPUDispatcher` returned by `njit` """
    if not isinstance(sig, Signature):
        raise ValueError(f"Expected a single signature, found {sig} of type {type(sig)}")

    def _(func):
        func_jit = njit(sig, **kwargs)(func)
        sigs = func_jit.nopython_signatures
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        func_cres = func_jit.get_compile_result(sigs[0])
        cres_wap = CompileResultWAP(func_cres)
        return cres_wap
    return _


def _cres_cacheable_global_name(func):
    """Stable cross-process LLVM-global symbol for ``func``'s entry-point address.

    Uses ``__qualname__`` (not just ``__name__``) so that nested helpers like
    ``outer.<locals>.inner`` or two methods of distinct classes named ``impl``
    in the same module each get a distinct symbol. The full qualname goes
    into the sha256 input (which is what guarantees injectivity); the
    plaintext part of the symbol embeds a sanitized qualname (``<`` and ``>``
    are not legal in unquoted LLVM identifiers, so we replace them with
    ``_``) so the symbol stays human-readable in ``nm`` / debugger output.
    ``.`` is a legal character in LLVM identifiers and in ELF / Mach-O /
    PE-COFF symbol tables.
    """
    fqn_full = f"{func.__module__}.{func.__qualname__}"
    fqn_safe = fqn_full.replace("<", "_").replace(">", "_")
    digest = hashlib.sha256(fqn_full.encode("utf-8")).hexdigest()[:16]
    return f"_numbox_cres_cacheable_addr_{fqn_safe}_{digest}"


def cres_cacheable(sig, **njit_kwargs):
    """Like ``cres``, but returns a ``CPUDispatcher`` that's cacheable from
    ``@njit(cache=True)`` callers.

    ``cres`` returns a ``CompileResultWAP`` (numba ``FunctionType`` value).
    When a user ``@njit(cache=True)`` references one as a Python global,
    numba's ``lower_constant_function_type`` materializes the value's
    runtime address via ``add_dynamic_addr``, which sets
    ``has_dynamic_globals`` on the caller's compiled IR and disables its
    persistent cache: ``NumbaWarning: Cannot cache compiled function "..."
    as it uses dynamic globals``.

    ``cres_cacheable`` produces two parallel artifacts:

    - An ``@cres`` ``CompileResultWAP``, attached as
      ``wrapped.as_funcvalue``. External consumers that need a
      ``FunctionType`` value (e.g. for storing in a struct field) still
      have one.
    - An ``@njit(sig, cache=True)`` dispatcher (this decorator's return
      value) that loads the underlying function pointer from a named LLVM
      global and indirect-calls through it. Cached caller IR references
      only the symbol name, not a runtime address — callers cache cleanly.

    Cross-process safety: a per-binding setup ``@njit(void(intp),
    cache=True)`` runs once per process at module import and writes the
    live ``CompileResultWAP`` address into the named global. The JIT
    linker resolves the symbol per process, so cached caller IR survives
    across processes and remains ASLR-correct.

    See ``numbox/utils/_addr_global.py`` for the underlying intrinsics and
    the CRE pattern they're adapted from.
    """
    if not isinstance(sig, Signature):
        raise ValueError(
            f"cres_cacheable: expected a single Signature, found "
            f"{sig!r} of type {type(sig).__name__}"
        )
    njit_kwargs.setdefault("cache", True)
    arg_tys = tuple(sig.args)
    n_args = len(arg_tys)
    ret_is_void = sig.return_type == void

    def _decorator(func):
        funcvalue = cres(sig, **njit_kwargs)(func)
        global_name = _cres_cacheable_global_name(func)
        _icall = _make_icall_for_sig(sig)
        fqn_stem = f"{func.__module__}.{func.__name__}"

        params = ", ".join(f"a{i}" for i in range(n_args))
        if n_args == 0:
            call_args = "_addr"
        elif n_args == 1:
            call_args = "_addr, a0"
        else:
            call_args = "_addr, (" + ", ".join(f"a{i}" for i in range(n_args)) + ")"
        body = (
            f"def _wrapper({params}):\n"
            f"    _addr = _load_addr({global_name!r})\n"
        )
        body += "    " + ("" if ret_is_void else "return ") + f"_icall({call_args})\n"

        # Anchor at a content-addressed real file (same defense as make_structref):
        # numba's cache-save path may call inspect.getsource on dynamically-exec'd
        # functions; on Python 3.13 a co_filename pointing at an unrelated real
        # file (highlevel.py) returns wrong content per CPython #122981.
        wrapper_anchor = _cres_cacheable_anchor_path(fqn_stem + ".wrapper", body)
        _materialize_anchor(wrapper_anchor, body)
        ns = {
            "_load_addr": _load_addr_from_named_global,
            "_icall": _icall,
        }
        exec(compile(body, str(wrapper_anchor), mode="exec"), ns)
        wrapper = ns["_wrapper"]
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = getattr(func, "__qualname__", func.__name__)
        wrapper.__module__ = func.__module__
        wrapper.__doc__ = func.__doc__

        compiled = njit(sig, **njit_kwargs)(wrapper)

        # One-time setup: write the live cres address into the named global.
        # The setup is itself @njit(cache=True) — cacheable because the addr
        # comes in as an argument (no Python global is baked into its IR) and
        # the global name is exec-inlined as a literal string.
        setup_body = (
            f"def _setup(addr):\n"
            f"    _store_addr({global_name!r}, addr)\n"
        )
        setup_anchor = _cres_cacheable_anchor_path(fqn_stem + ".setup", setup_body)
        _materialize_anchor(setup_anchor, setup_body)
        setup_ns = {"_store_addr": _store_addr_to_named_global}
        exec(compile(setup_body, str(setup_anchor), mode="exec"), setup_ns)
        _setup_py = setup_ns["_setup"]
        _setup_py.__name__ = f"_setup_for_{func.__name__}"
        _setup_py.__qualname__ = _setup_py.__name__
        _setup_py.__module__ = __name__
        _setup = njit(void(intp), **njit_kwargs)(_setup_py)
        _setup(funcvalue.address)

        compiled.as_funcvalue = funcvalue
        return compiled

    return _decorator


def cres_if_available(lib, sig, **kwargs):
    """Like ``cres(sig, **kwargs)``, but stubs out the wrapper if the C
    symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols may only exist in newer releases. Callers get a stub that
    raises ``NotImplementedError`` instead of a confusing LLVM link error
    at call time.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return cres(sig, **kwargs)(func)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{func.__name__} is not available")
        stub.__name__ = func.__name__
        return stub
    return _


def determine_field_index(struct_ty, field_name):
    for i_, field_pair in enumerate(struct_ty._fields):
        if field_pair[0] == field_name:
            return i_
    raise ValueError(f"{field_name} not in {struct_ty}")


def hash_type(ty: Type):
    mangled_ty = mangle_type_or_value(ty)
    return hashlib.sha256(mangled_ty.encode("utf-8")).hexdigest()


def make_structref_code_txt(
    struct_name: str,
    struct_fields: Iterable[str] | dict[str, Type],
    struct_type_class: type | Type,
    struct_methods: Optional[dict[str, Callable]] = None
):
    if isinstance(struct_fields, dict):
        struct_fields, fields_types = list(struct_fields.keys()), list(struct_fields.values())
    else:
        assert isinstance(struct_fields, (list, tuple)), struct_fields
        fields_types = None
    struct_fields_str = ", ".join([field for field in struct_fields])
    make_name = f"make_{struct_name.lower()}"
    new_returns = f"{make_name}({struct_fields_str})"
    repr_str = f"f'{struct_name}(" + ", ".join([f"{field}={{self.{field}}}" for field in struct_fields]) + ")'"
    code_txt = StringIO()
    code_txt.write(f"""
class {struct_name}(StructRefProxy):
    def __new__(cls, {struct_fields_str}):
        return {new_returns}

    def __repr__(self):
        return {repr_str}
""")
    for field in struct_fields:
        code_txt.write(f"""
    @property
    @njit(**jit_options)
    def {field}(self):
        return self.{field}
""")
    methods_code_txt = StringIO()
    if struct_methods is not None:
        assert isinstance(struct_methods, dict), f"""
    Expected dictionary of methods names to callable, got {struct_methods}"""
        for method_name, method in struct_methods.items():
            params_str, names_params_str = make_params_strings(method)
            names_params_lst = names_params_str.split(", ")
            self_name = names_params_lst[0]
            names_params_str_wo_self = ", ".join(names_params_lst[1:])
            method_source = dedent(getsource(method))
            method_hash = hashlib.sha256(method_source.encode("utf-8")).hexdigest()
            code_txt.write(f"""
    def {method_name}({params_str}):
        return {self_name}.{method_name}_{method_hash}({names_params_str_wo_self})
    
    @njit(**jit_options)
    def {method_name}_{method_hash}({params_str}):
        return {self_name}.{method_name}({names_params_str_wo_self})
""")
            method_header = re.findall(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*:[^\n]*", method_source, re.MULTILINE)
            assert len(method_header) == 1, method_header
            method_name, params_str_ = method_header[0]
            assert params_str == params_str_, (params_str, params_str_)
            method_source = re.sub(r"\bdef\s+([a-zA-Z_]\w*)\b", f"def _", method_source)
            methods_code_txt.write(f"""
@overload_method({struct_type_class.__name__}, "{method_name}", jit_options=jit_options)
def ol_{method_name}({params_str}):
{indent(method_source, "    ")}
    return _
""")
    code_txt.write(f"""
define_boxing({struct_type_class.__name__}, {struct_name})
""")
    struct_type_name = f"{struct_name}Type"
    struct_fields_ty_str = ", ".join([f"{field}_ty" for field in struct_fields])
    struct_type_code_block = ""
    if fields_types is None:
        struct_type_code_block = f"""fields_types = [{struct_fields_ty_str}]
    fields_and_their_types = list(zip(fields, fields_types))    
    {struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)        
"""
    else:
        code_txt.write(f"""
fields_and_their_types = list(zip(fields, fields_types))    
{struct_name}Type = {struct_type_class.__name__}(fields_and_their_types)
""")
    ctor_code_block = "\n".join([f"        struct_.{field} = {field}" for field in struct_fields])
    code_txt.write(f"""
@overload({struct_name}, strict=False, jit_options=jit_options)
def ol_{struct_name.lower()}({struct_fields_ty_str}):
    {struct_type_code_block}
    def ctor({struct_fields_str}):
        struct_ = new({struct_type_name})
{ctor_code_block}
        return struct_
    return ctor
""")
    if fields_types is not None:
        code_txt.write(f"""
{make_name}_sig = {struct_name}Type(*fields_types)
""")
    else:
        code_txt.write(f"""
{make_name}_sig = None
""")
    code_txt.write(f"""
@njit({make_name}_sig, **jit_options)
def {make_name}({struct_fields_str}):
    return {struct_name}({struct_fields_str})
""")
    code_txt = code_txt.getvalue() + methods_code_txt.getvalue()
    return code_txt, fields_types


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


def _anchor_path(subdir: str, stem: str, code_txt: str) -> Path:
    """Stable on-disk source anchor for dynamically-exec'd code.

    Content-addressed so identical generated text always resolves to the
    same path; numba's source-mtime cache key then hits across runs and
    processes. The anchor is a real file whose contents match the exec'd
    code line-for-line, so `inspect.getsource` returns the right source.
    This sidesteps `CPython #122981
    <https://github.com/python/cpython/issues/122981>`_: on Python 3.13,
    ``inspect.getsource`` on an exec'd function whose ``co_filename``
    anchors at an unrelated real file returns arbitrary other content.
    """
    root = _anchor_root(subdir)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return root / f"{stem}_{digest}.py"


def _structref_anchor_path(struct_name: str, code_txt: str) -> Path:
    return _anchor_path("numbox-structref", struct_name, code_txt)


def _cres_cacheable_anchor_path(stem: str, code_txt: str) -> Path:
    return _anchor_path("numbox-cres-cacheable", stem, code_txt)


def _materialize_anchor(path: Path, code_txt: str) -> None:
    if path.exists():
        return
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code_txt)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _orphan_anchor_sweep(subdir: str) -> None:
    """Best-effort cleanup of orphaned ``.tmp-*`` anchor files from SIGKILL'd
    writers. Called at module import; failures are silent (the orphan is at
    worst harmless disk usage).
    """
    try:
        root = _anchor_root(subdir)
        if not root.exists():
            return
        for orphan in root.glob("*.tmp-*"):
            try:
                orphan.unlink()
            except OSError:
                pass
    except Exception:
        pass


_orphan_anchor_sweep("numbox-structref")
_orphan_anchor_sweep("numbox-cres-cacheable")


def make_structref(
    struct_name: str,
    struct_fields: Iterable[str] | dict[str, Type],
    struct_type_class: type | Type,
    *,
    struct_methods: Optional[dict[str, Callable]] = None,
    jit_options: Optional[dict] = None
):
    """
    Makes structure type with `struct_name` and `struct_fields` from the StructRef type class.

    A unique `struct_type_class` for each structref needs to be provided.
    If caching of code that will be using the created struct type is desired,
    these type class(es) need/s to be defined in a python module that is *not* executed.
    (Same requirement is also to observed even when the full definition of StructRef
    is entirely hard-coded rather than created dynamically.)

    In particular, that's why `struct_type_class` cannot be incorporated into
    the dynamic compile / exec routine here.

    Dictionary of methods to be bound to the created structref can be provided as well.
    Struct methods will get inlined into the caller if numba deems it to be optimal
    (even if `jit_options` says otherwise), therefore changing the methods code
    without poking the jitted caller can result in a stale cache - when the latter is
    cached. This is not an exclusive limitation of a dynamic structref creation via
    this function and is equally true when the structref definition is coded explicitly.

    Anchor file (Python 3.13 / numba interaction — important)
    --------------------------------------------------------
    The generated ``code_txt`` is written to a content-addressed file under
    numba's cache directory and that file — not ``highlevel.py`` — is used as
    the ``compile()`` anchor. Without this, numba's cache-save path calls
    ``inspect.getsourcelines`` on the generated functions, which reads
    arbitrary content from ``highlevel.py`` at the recorded ``co_firstlineno``;
    on Python 3.13 ``inspect.getsource`` returns that wrong content (`CPython
    #122981 <https://github.com/python/cpython/issues/122981>`_), and numba's
    cache-save chokes on it. See ``_structref_anchor_path`` for the path scheme.
    """
    code_txt, fields_types = make_structref_code_txt(
        struct_name, struct_fields, struct_type_class, struct_methods
    )
    if jit_options is None:
        jit_options = default_jit_options
    ns = {
        **getmodule(_file_anchor).__dict__,
        **{
            "fields": struct_fields,
            "fields_types": fields_types,
            "define_boxing": define_boxing,
            "jit_options": jit_options,
            "new": new,
            "njit": njit,
            "overload": overload,
            "overload_method": overload_method,
            "StructRefProxy": StructRefProxy,
            struct_type_class.__name__: struct_type_class
        }
    }
    anchor = _structref_anchor_path(struct_name, code_txt)
    _materialize_anchor(anchor, code_txt)
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)
    return ns[struct_name]


def prune_type(ty):
    if isinstance(ty, Dispatcher):
        sigs = ty.get_call_signatures()[0]
        assert len(sigs) == 1, f"Ambiguous signature, {sigs}"
        ty = FunctionType(sigs[0])
    return ty
