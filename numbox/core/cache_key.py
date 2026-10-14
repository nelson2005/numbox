"""numbox's part of numba's on-disk cache key.

numba keys a cached function on its signature, the target machine, and hashes of its bytecode and closure. The jit
flags it compiled under, the ``NUMBA_*`` settings that change the code it emits, the values of the globals it reads
and the code of the functions it calls are not in that key, so a change to any of them can be served a binary built
under the old one. numbox adds all of them, for every function it compiles and for any function decorated with
:func:`keyed_njit` or :func:`keyed_cfunc`; numba keeps its own key for everything else.

What the key holds, past numba's own:

- **numbox's version.** numbox's ``@overload`` and ``@intrinsic`` code generators shape the code of whatever calls
  them, and the key cannot see into them.
- **The flags.** The decorator kind (``njit`` or ``cfunc``), its options, its ``locals`` and its pipeline class.
- **The settings.** Each of :data:`KEYED_SETTINGS`, twice: the value it had when numba built its code generator, and
  its value now. numba reads some of them once, when it builds its code generator, and others at every compile, and
  which is which differs by setting and by numba version, so both are keyed.
- **The code and the values it reads, all the way down.** The bytecode, constants, default arguments and closure
  cells of the function, the value of every global it reads, the module attributes it reads, and the same for every
  function it calls, followed through globals, closures and module attributes. An ``@intrinsic``, a numba type, a
  class, a ufunc, a builtin, a callable numba types through a registered ``@overload`` and a context numba compiles a
  ``with`` block over (``objmode``) go in by name, with the version of the package that defines them; arrays are
  hashed in full, and a namedtuple goes in with its field names. A class whose contents numba compiles goes in with
  them: a namedtuple class's field names, an ``Enum`` class's member values, a jitclass's field types and methods.

Each of those is its own part of the key, so :func:`explain_cache_key` can list them and the churn warning can name
the ones that keep changing. A value with no stable form, which no process could put in the key the same way twice,
compiles the function without a cache, with an :class:`UncachedWarning` naming it. So does a function it calls that
numba caches under its own key alone, a plain ``numba.njit(cache=True)``: numba loads that function from its own
cache whatever changed in what it reads, and the keyed function would hold the old code under a new key.

The values are read when numba looks the function up, so a setting or global changed in code after import counts for
anything compiled afterwards. A ``NUMBA_*`` variable set in ``os.environ`` counts too: numba reads those again as a
compile starts, and the key reads them as that compile will. numba keeps a function it has already compiled for the
life of the process, whatever changes afterwards: that is numba's behaviour, and the key on disk names what the binary
was built under.

numba reads a few settings when it builds its code generator, which it does at the first compile in a process. numbox
records them then, as long as numbox was imported first; where numba compiled something before numbox was imported,
the values at numbox's import stand in for them, so a setting changed in code between that first compile and numbox's
import is not seen.

An edit to a package's code generators in place, a development checkout's ``@intrinsic`` or ``@overload``, changes
nothing the key can see: clear the cache after one. One of your own code, from a module no installed package provides,
has no version to stand in for it and is followed like any function, so an edit to it or to a value it reads misses.
"""
import contextlib
import enum
import functools
import hashlib
import importlib.metadata
import sys
import threading
import warnings

from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, ModuleType

import numpy as np

from llvmlite import ir as llvm_ir
from numba import cfunc, njit
from numba.core import caching, codegen as numba_codegen, config as numba_config, registry
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.extending import _Intrinsic
from numba.core.types import ExternalFunction, Type as NumbaType
from numba.core.types.function_type import CompileResultWAP
from numba.core.types.misc import ClassInstanceType, ClassType
from numba.core.typing.typeof import Purpose, typeof
from numba.core.withcontexts import WithContext
from numba.np.ufunc.dufunc import DUFunc

import numbox
from numbox.utils.fingerprint import (
    _ABSENT_ALIASES, _Unfingerprintable, _canon_value, _dtype_key, _fingerprint_codeobj, _loaded_global_names,
    _namedtuple_class, _referenced_global_names,
)


#: The ``NUMBA_*`` settings that change the code numba emits, each with the ``numba.config`` attribute numba reads it
#: into. Classified by a survey of numba 0.60 to 0.68 (each minor at its latest patch), which compiled a fixed corpus
#: under every setting flipped and compared the emitted LLVM IR and assembly: a setting is keyed if flipping it
#: changes the code, speed-only and debug-information settings included. ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and
#: ``NUMBA_DIFF_IR`` change the code only as a side effect of the dumps they ask for, and are keyed all the same. Three
#: change it only beside another, which a flip of one setting at a time cannot show, and were found in numba's source:
#: ``NUMBA_DEBUG_NRT_STACK_LIMIT`` under ``NUMBA_DEBUG_NRT``, and ``NUMBA_FULL_TRACEBACKS`` and ``NUMBA_DEVELOPER_MODE``
#: (its default) in a bounds check.
KEYED_SETTINGS = {
    "NUMBA_BOUNDSCHECK": "BOUNDSCHECK",
    "NUMBA_OPT": "OPT",
    "NUMBA_LOOP_VECTORIZE": "LOOP_VECTORIZE",
    "NUMBA_SLP_VECTORIZE": "SLP_VECTORIZE",
    "NUMBA_ENABLE_AVX": "ENABLE_AVX",
    "NUMBA_DEBUGINFO": "DEBUGINFO_DEFAULT",
    "NUMBA_ENABLE_PROFILING": "ENABLE_PROFILING",
    "NUMBA_EXTEND_VARIABLE_LIFETIMES": "EXTEND_VARIABLE_LIFETIMES",
    "NUMBA_LLVM_REFPRUNE_PASS": "LLVM_REFPRUNE_PASS",
    "NUMBA_LLVM_REFPRUNE_FLAGS": "LLVM_REFPRUNE_FLAGS",
    "NUMBA_DEBUG_NRT": "DEBUG_NRT",
    "NUMBA_DEBUG_NRT_STACK_LIMIT": "DEBUG_NRT_STACK_LIMIT",
    "NUMBA_FULL_TRACEBACKS": "FULL_TRACEBACKS",
    "NUMBA_DEVELOPER_MODE": "DEVELOPER_MODE",
    "NUMBA_DEBUG_JIT": "DEBUG_JIT",
    "NUMBA_DEBUG_ARRAY_OPT": "DEBUG_ARRAY_OPT",
    "NUMBA_DEBUG_ARRAY_OPT_RUNTIME": "DEBUG_ARRAY_OPT_RUNTIME",
    "NUMBA_PARFOR_MAX_TUPLE_SIZE": "PARFOR_MAX_TUPLE_SIZE",
    "NUMBA_USE_RVSDG_FRONTEND": "USE_RVSDG_FRONTEND",
    "NUMBA_GDB_BINARY": "GDB_BINARY",
    "NUMBA_DUMP_ASSEMBLY": "DUMP_ASSEMBLY",
    "NUMBA_DEBUG": "DEBUG",
    "NUMBA_DIFF_IR": "DIFF_IR",
}

_IN_NUMBAS_KEY = "in numba's own key, through the target machine"
_NO_BINARY = "produces no binary"
_SVML = "left out: SVML is outside what numbox keys"
_NO_CHANGE = "changes no emitted code"
_LOCATION = "chooses where the cache lives, not what goes in it"

#: Every other ``NUMBA_*`` setting numba 0.60 to 0.68 reads, with the reason it stays out of the key. A test fails when
#: a numba release reads a setting that is in neither table.
UNKEYED_SETTINGS = {
    "NUMBA_CPU_NAME": _IN_NUMBAS_KEY,
    "NUMBA_CPU_FEATURES": _IN_NUMBAS_KEY,
    "NUMBA_DISABLE_JIT": _NO_BINARY,
    "NUMBA_USE_LEGACY_TYPE_SYSTEM": _NO_BINARY,
    "NUMBA_DISABLE_INTEL_SVML": _SVML,
    "NUMBA_CACHE_DIR": _LOCATION,
    "NUMBA_CACHE_LOCATOR_CLASSES": _LOCATION,
    **dict.fromkeys((
        "NUMBA_ALWAYS_WARN_UNINIT_VAR", "NUMBA_CAPTURED_ERRORS", "NUMBA_CHROME_TRACE", "NUMBA_COLOR_SCHEME",
        "NUMBA_DEBUG_ARRAY_OPT_STATS", "NUMBA_DEBUG_CACHE", "NUMBA_DEBUG_FRONTEND", "NUMBA_DEBUG_INLINE_CLOSURE",
        "NUMBA_DEBUG_PRINT_AFTER", "NUMBA_DEBUG_PRINT_BEFORE", "NUMBA_DEBUG_PRINT_WRAP",
        "NUMBA_DEBUG_TYPEINFER", "NUMBA_DISABLE_CUDA", "NUMBA_DISABLE_PERFORMANCE_WARNINGS",
        "NUMBA_DISABLE_TYPEINFER_FAIL_CACHE", "NUMBA_DUMP_ANNOTATION", "NUMBA_DUMP_BYTECODE", "NUMBA_DUMP_CFG",
        "NUMBA_DUMP_FUNC_OPT", "NUMBA_DUMP_HTML", "NUMBA_DUMP_IR", "NUMBA_DUMP_LLVM", "NUMBA_DUMP_OPTIMIZED",
        "NUMBA_DUMP_SSA", "NUMBA_ENABLE_CUDASIM", "NUMBA_ENABLE_SYS_MONITORING", "NUMBA_FORCE_CUDA_CC",
        "NUMBA_FUNCTION_CACHE_SIZE", "NUMBA_HIGHLIGHT_DUMPS", "NUMBA_JIT_COVERAGE",
        "NUMBA_LLVM_PASS_TIMINGS", "NUMBA_NRT_STATS", "NUMBA_NUM_THREADS", "NUMBA_PARALLEL_DIAGNOSTICS",
        "NUMBA_SHOW_HELP", "NUMBA_THREADING_LAYER", "NUMBA_THREADING_LAYER_PRIORITY", "NUMBA_TRACE",
        "NUMBA_USE_LLVMLITE_MEMORY_MANAGER",
        "NUMBA_CUDA_ARRAY_INTERFACE_SYNC", "NUMBA_CUDA_DEBUGINFO", "NUMBA_CUDA_DEFAULT_PTX_CC", "NUMBA_CUDA_DRIVER",
        "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY", "NUMBA_CUDA_INCLUDE_PATH", "NUMBA_CUDA_LOG_API_ARGS",
        "NUMBA_CUDA_LOG_LEVEL", "NUMBA_CUDA_LOG_SIZE", "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS",
        "NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT", "NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO", "NUMBA_CUDA_MEMORY_MANAGER",
        "NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM", "NUMBA_CUDA_USE_NVIDIA_BINDING", "NUMBA_CUDA_VERBOSE_JIT_LOG",
        "NUMBA_CUDA_WARN_ON_IMPLICIT_COPY",
    ), _NO_CHANGE),
}

#: An index holding more entries than this for one function draws a :class:`CacheChurnWarning` naming the parts of the
#: key that differ between them.
CHURN_THRESHOLD = 100


class UncachedWarning(RuntimeWarning):
    """A function numbox keys is compiled without an on-disk cache: a value it reads has no stable form for the key,
    or a function it calls is cached under numba's key alone."""


class CacheChurnWarning(RuntimeWarning):
    """A function's cache holds an entry per value of something it reads that changes between runs."""


_NO_STABLE_FORM = (
    "numbox keys the cache on the values a function reads, and a value no process can put in the key the same way "
    "twice would serve one process's binary to another."
)
_CACHED_ALONE = (
    "numba loads such a function from its own cache, which a change to what it reads does not miss, so this "
    "function's new binary would hold the old code under a key naming the new value."
)


class _Unkeyable(Exception):
    """Raised inside the key computation for what the key cannot hold; the cache wrappers turn it into a miss.

    ``why`` is what the warning says about it, a value with no stable form unless given.
    """

    def __init__(self, message, why=_NO_STABLE_FORM):
        super().__init__(message)
        self.why = why


def _cached_under_numbas_key_alone(cache) -> bool:
    """Whether ``cache`` is an on-disk cache numbox does not key, as a plain ``numba.njit(cache=True)``'s is."""
    return cache is not None and not isinstance(cache, caching.NullCache) and getattr(cache, "_numbox_keyed", None) is None


def _setting_value(value) -> str:
    """The canonical text of one setting's value.

    numba's ``NUMBA_OPT`` is an ``int`` that keeps the value it was given, ``max`` among them, in ``_raw_value``; a
    bool and an int that agree mean the same to numba, and are put the same way.
    """
    raw = getattr(value, "_raw_value", None)
    if raw is not None:
        value = raw
    if isinstance(value, int):
        value = int(value)
    return repr(value)


def _current_settings() -> tuple:
    # numba re-reads its NUMBA_* environment variables when a compile starts, which is after the cache lookup: read
    # them as that compile will, or one set in os.environ after import keys the lookup and the save apart.
    numba_config.reload_config()
    return tuple((env, _setting_value(getattr(numba_config, attr, None))) for env, attr in KEYED_SETTINGS.items())


def _settings_at_build(codegen) -> tuple:
    """The settings ``codegen`` was built under, as numbox recorded them, or the values now if it recorded none."""
    return getattr(codegen, "_numbox_settings_at_build", None) or _current_settings()


def _settings_text(codegen) -> str:
    built, now = _settings_at_build(codegen), _current_settings()
    return repr([(env, at_build, value) for (env, at_build), (_, value) in zip(built, now)])


def _settings_canon() -> str:
    """The settings part of the key as it stands for the next compile in this process.

    Shared with the content-addressed names of the kernels :mod:`numbox.core.variable.compile_kernel` generates and
    the derives :mod:`numbox.core.work.builder` anchors, which fold it into their digests.
    """
    return _settings_text(registry.cpu_target.target_context.codegen())


_packages = None


def _package_version(module_name) -> str:
    """``<distribution>==<version>`` for the installed distribution that provides ``module_name``, else ``-``.

    The top-level package's ``__version__`` answers first, since most packages carry one and reading it costs nothing;
    the distributions on the path answer for the rest. A module of no installed distribution, a script or a notebook,
    has no version to give.
    """
    top = (module_name or "").partition(".")[0]
    if not top:
        return "-"
    if top == "numbox":
        return f"numbox=={numbox.__version__}"
    return _distribution_version(top)


@functools.lru_cache(maxsize=None)
def _distribution_version(top) -> str:
    version = getattr(sys.modules.get(top), "__version__", None)
    if isinstance(version, str):
        return f"{top}=={version}"
    global _packages
    if _packages is None:
        try:
            _packages = importlib.metadata.packages_distributions()
        except Exception:
            _packages = {}
    for distribution in _packages.get(top, ()):
        try:
            return f"{distribution}=={importlib.metadata.version(distribution)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return "-"


def _own_code(func) -> bool:
    """Whether ``func`` is code of your own: written in a module no installed package provides, so no version stands
    in for it, and reading that module's namespace. Code numbox generates runs in a namespace of its own, though it may
    carry the calling module's name, and is numbox's, under numbox's version."""
    module = getattr(func, "__module__", None)
    if _package_version(module) != "-":
        return False
    return getattr(sys.modules.get(module), "__dict__", None) is func.__globals__


def _is_register_jitable(template) -> bool:
    """Whether numba made ``template`` for a ``register_jitable`` function, whose own body is what numba compiles."""
    overload_func = getattr(template, "_overload_func", None)
    return getattr(overload_func, "__qualname__", "").startswith("register_jitable.")


class _Walk:
    """One computation of a function's parts of the key.

    ``parts`` maps the name of each part to its canonical text: ``code <function>`` for a function's code, defaults and
    keyword defaults; ``closure <function>.<cell>`` for a closure cell; ``global <module>.<name>`` for a global or a
    module attribute. A function is named by its module and qualified name; two different functions of one name met in
    one walk, a factory's closures, are told apart by a suffix given in the order the walk meets them, which is the
    same in every process.
    """

    def __init__(self):
        self.parts = {}
        self._functions = {}
        self._names = {}
        self._classes = {}
        self._typing = None

    def put(self, name, text):
        candidate, n = name, 1
        while candidate in self.parts and self.parts[candidate] != text:
            n += 1
            candidate = f"{name}#{n}"
        self.parts[candidate] = text

    def function(self, func) -> str:
        """Walk ``func`` into the parts, once, and return the text that refers to it."""
        ref = self._functions.get(id(func))
        if ref is not None:
            return ref
        name = f"{func.__module__}.{func.__qualname__}"
        count = self._names[name] = self._names.get(name, 0) + 1
        if count > 1:
            name = f"{name}#{count}"
        ref = self._functions[id(func)] = f"function({name})"
        code = func.__code__
        where = f"the function {func.__qualname__}"
        self.parts[f"code {name}"] = (
            f"{_fingerprint_codeobj(code, set())};"
            f"defaults={self.value(func.__defaults__ or (), f'a default argument of {where}')};"
            f"kwdefaults={self.value(func.__kwdefaults__ or {}, f'a keyword default of {where}')}"
        )
        closure = {}
        for cell_name, cell in zip(code.co_freevars, func.__closure__ or ()):
            try:
                contents = cell.cell_contents
            except ValueError:
                raise _Unkeyable(f"the closure cell {cell_name} of {where} is empty") from None
            closure[cell_name] = contents
            self.put(f"closure {name}.{cell_name}", self.value(contents, f"the closure cell {cell_name} of {where}"))
        namespace = func.__globals__
        for global_name in sorted(_loaded_global_names(code)):
            if global_name in namespace:
                self.put(
                    f"global {func.__module__}.{global_name}",
                    self.value(namespace[global_name], f"the global {global_name} read by {where}"),
                )
        referenced = _referenced_global_names(code)
        reachable = {n: namespace[n] for n in referenced if n in namespace}
        reachable.update(closure)
        self.module_attributes(referenced, reachable, where)
        return ref

    def module_attributes(self, referenced, reachable, reader):
        """Fold the ``<module>.<attribute>`` reads numba freezes, for every module ``reachable`` names.

        The names a function's code refers to pool attribute names with global ones, so an attribute is folded when a
        module reachable from the function has one of those names, as a global of that name would be: a
        ``register_jitable`` function is followed, an ``@intrinsic`` or a dtype goes in as it would by name. The name
        may be some other object's attribute that the function never reads from the module, so a value numba could not
        compile is passed over, and leaves the walk as it found it: a plain function numba holds no typing for, or a
        value with no stable form that numba cannot type either. A value with no stable form that numba can compile, an
        ``Enum`` member say, is refused, as it is read by name. The module's ``__dict__`` is read, never ``getattr``,
        so no module's ``__getattr__`` runs.
        """
        modules_seen = set()

        def walk(module):
            if id(module) in modules_seen:
                return
            modules_seen.add(id(module))
            module_dict = getattr(module, "__dict__", {})
            for attribute in sorted(referenced):
                if attribute not in module_dict:
                    continue
                value = module_dict[attribute]
                if isinstance(value, ModuleType):
                    walk(value)
                    continue
                if isinstance(value, FunctionType) and self.templates(value) is None:
                    continue
                state = dict(self.parts), dict(self._functions), dict(self._names), dict(self._classes)
                try:
                    text = self.value(value, f"the module attribute {module.__name__}.{attribute} read by {reader}")
                except _Unkeyable:
                    if self.compiles(value):
                        raise
                    self.parts, self._functions, self._names, self._classes = state
                    continue
                self.put(f"global {module.__name__}.{attribute}", text)

        for name in sorted(reachable):
            if isinstance(reachable[name], ModuleType):
                walk(reachable[name])

    def flags(self, kind, compiler) -> str:
        targetoptions = dict(getattr(compiler, "targetoptions", None) or {})
        locals_ = dict(getattr(compiler, "locals", None) or {})
        pipeline = getattr(compiler, "pipeline_class", None)
        return (
            f"kind={kind};targetoptions={self.value(targetoptions, 'a jit option')};"
            f"locals={self.value(locals_, 'a type in locals')};pipeline={self.value(pipeline, 'the pipeline class')}"
        )

    def templates(self, func):
        """The typing templates numba holds for ``func``, or ``None`` where it holds no typing for it."""
        if self._typing is None:
            self._typing = registry.cpu_target.typing_context
            self._typing.refresh()
        try:
            function_type = self._typing._lookup_global(func)
        except KeyError:
            return None
        if function_type is None:
            return None
        return tuple(getattr(function_type, "templates", ()))

    def registered(self, func) -> bool:
        """Whether numba types ``func`` through a registered template, and so never compiles its body."""
        templates = self.templates(func)
        return templates is not None and not any(_is_register_jitable(t) for t in templates)

    def compiles(self, value) -> bool:
        """Whether numba compiles ``value`` into a function that reads it: code it calls, a callable it holds typing
        for, or a constant it can type."""
        if isinstance(value, (Dispatcher, CFunc, DUFunc, CompileResultWAP, _Intrinsic, FunctionType)):
            return True
        if self.templates(value) is not None:
            return True
        try:
            typeof(value, Purpose.constant)
        except Exception:
            return False
        return True

    def intrinsic(self, value) -> str:
        """An ``@intrinsic``, by name and its package's version, with its definition's code and closure.

        What an intrinsic emits is decided by the code it runs at compile time, which the key does not follow past an
        installed package's version. Its own definition's code and closure are folded all the same, which tells apart
        the intrinsics a factory makes, and the ones ``@proxy`` generates, which name the body they call in a
        constant. Its code generator reads its closure as it compiles, so a closure cell with no stable form refuses
        the key, as a global read by name does. An intrinsic of code no installed package provides has no version to
        stand in for its code generator, so its definition is walked like any function, the values it reads with it.
        """
        defn = getattr(value, "_defn", None)
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None) or getattr(value, "_name", "?")
        if isinstance(defn, FunctionType) and _own_code(defn):
            return f"intrinsic({module}.{qualname};-;{self.function(defn)})"
        body, cells = "-", []
        if isinstance(defn, FunctionType):
            body = _fingerprint_codeobj(defn.__code__, set())
            for cell_name, cell in zip(defn.__code__.co_freevars, defn.__closure__ or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    cells.append(f"{cell_name}=<empty>")
                    continue
                where = f"the closure cell {cell_name} of the intrinsic {qualname}"
                cells.append(f"{cell_name}={self.value(contents, where)}")
        return f"intrinsic({module}.{qualname};{_package_version(module)};{body};closure=[{';'.join(cells)}])"

    def overloaded(self, func) -> str:
        """A callable numba types through a registered ``@overload``, by name and its package's version.

        numba compiles what the ``@overload`` returns, never the callable's own body. An overload of code no installed
        package provides has no version to stand in for it, so its function is walked like any function, the
        implementation it returns and the values that reads with it.
        """
        module = getattr(func, "__module__", None)
        qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", "?")
        own = []
        for template in self.templates(func) or ():
            overload_func = getattr(template, "_overload_func", None)
            if isinstance(overload_func, FunctionType) and _own_code(overload_func):
                own.append(self.function(overload_func))
        walked = f";overloads=[{','.join(own)}]" if own else ""
        return f"overloaded({module}.{qualname};{_package_version(module)}{walked})"

    def signature(self, sig, where) -> str:
        """A signature by its types, each as the key holds it: numba's own text for one names a jitclass's types after
        an address no other process shares."""
        args = ",".join(self.value(ty, where) for ty in sig.args)
        return f"({args})->{self.value(sig.return_type, where)}"

    def jitclass(self, class_type, where) -> str:
        """A jitclass by its fields and the code of its methods, which numba compiles into the code that uses it.

        numba names a jitclass's types after the address of its ``ClassType``, which no other process shares, so that
        name stays out of the key: the jitclass is referred to by its class name and the order the walk meets it in.
        """
        ref = self._classes.get(id(class_type))
        if ref is not None:
            return ref
        ref = self._classes[id(class_type)] = f"jitclass({class_type.class_name}#{len(self._classes) + 1})"
        fields = ",".join(f"{name}:{self.value(ty, where)}" for name, ty in class_type.struct.items())
        methods = self.value(dict(class_type.jit_methods), where)
        properties = self.value({name: dict(accessors) for name, accessors in class_type.jit_props.items()}, where)
        static = self.value(dict(class_type.jit_static_methods), where)
        self.put(
            f"jitclass {class_type.class_name}",
            f"{ref};fields=[{fields}];methods={methods};properties={properties};static={static}",
        )
        return ref

    def value(self, value, where) -> str:
        """The canonical text of ``value``, walking any function it holds into the parts."""
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
            return repr(value)
        if isinstance(value, (tuple, list)):
            head = _namedtuple_class(type(value)) or type(value).__name__
            return f"{head}[" + ",".join(self.value(v, where) for v in value) + "]"
        if isinstance(value, (set, frozenset)):
            return f"{type(value).__name__}[" + ",".join(sorted(self.value(v, where) for v in value)) + "]"
        if isinstance(value, dict):
            items = sorted((self.value(k, where), self.value(v, where)) for k, v in value.items())
            return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
        if isinstance(value, Dispatcher):
            alias = getattr(value, "_numbox_proxy_alias", None)
            if alias is not None and alias not in _ABSENT_ALIASES:
                # A @proxy binding's alias names its body, signature and options; its own wrapper calls an
                # intrinsic that only resolves the alias.
                return f"proxy({alias})"
            if _cached_under_numbas_key_alone(getattr(value, "_cache", None)):
                raise _Unkeyable(
                    f"{where} is cached under numba's key alone; decorate it with keyed_njit, or drop its cache=True",
                    why=_CACHED_ALONE,
                )
            return f"dispatcher({self.function(value.py_func)};{self.flags('njit', value._compiler)})"
        if isinstance(value, CFunc):
            return f"cfunc({self.function(value._pyfunc)};{self.flags('cfunc', value._compiler)})"
        if isinstance(value, DUFunc):
            wrapped = getattr(value, "__wrapped__", None)
            if not isinstance(wrapped, FunctionType):
                raise _Unkeyable(f"{where} is a @vectorize ufunc whose body cannot be read")
            if _cached_under_numbas_key_alone(getattr(getattr(value, "_dispatcher", None), "cache", None)):
                raise _Unkeyable(
                    f"{where} is a @vectorize ufunc cached under numba's key alone; drop its cache=True",
                    why=_CACHED_ALONE,
                )
            targetoptions = self.value(dict(getattr(value, "targetoptions", None) or {}), where)
            return f"dufunc({self.function(wrapped)};{targetoptions})"
        if isinstance(value, CompileResultWAP):
            alias = getattr(value, "_numbox_proxy_alias", None)
            if alias is not None and alias not in _ABSENT_ALIASES:
                return f"proxyfunc({alias})"
            py_func = getattr(value, "_numbox_py_func", None)
            if py_func is None:
                raise _Unkeyable(f"{where} is a first-class function value numbox did not make")
            return f"cresfunc({self.function(py_func)};{self.signature(value.cres.signature, where)})"
        if isinstance(value, _Intrinsic):
            return self.intrinsic(value)
        if isinstance(value, FunctionType):
            if self.registered(value):
                return self.overloaded(value)
            return self.function(value)
        if isinstance(value, ExternalFunction):
            # numba knows it by its symbol and signature; its name holds the symbol alone.
            return f"externalfunction({value.symbol};{self.signature(value.sig, where)})"
        if isinstance(value, ClassInstanceType):
            return f"instance({self.jitclass(value.class_type, where)})"
        if isinstance(value, NumbaType):
            text = str(value)
            if "0x" in text or "jitclass." in text or "DeferredType#" in text:
                # A jitclass's types and a deferred type are named after the address of an object of numba's.
                raise _Unkeyable(f"{where} is the numba type {text}, which names an object by its address")
            module = type(value).__module__
            return f"numbatype({module}.{type(value).__qualname__}:{text};{_package_version(module)})"
        if isinstance(value, type):
            class_type = vars(value).get("class_type")
            if isinstance(class_type, ClassType):
                return self.jitclass(class_type, where)
            name = _namedtuple_class(value) or f"{value.__module__}.{value.__qualname__}"
            if issubclass(value, enum.Enum):
                # numba compiles a read of a member to the member's value.
                members = ",".join(f"{n}={self.value(m.value, where)}" for n, m in value.__members__.items())
                name = f"{name};members=[{members}]"
            return f"class({name};{_package_version(value.__module__)})"
        if isinstance(value, np.ufunc):
            module = getattr(value, "__module__", None) or "numpy"
            return f"ufunc({module}.{value.__name__};{_package_version(module)})"
        if isinstance(value, (BuiltinFunctionType, BuiltinMethodType)):
            module = getattr(value, "__module__", None)
            return f"builtin({module}.{value.__qualname__};{_package_version(module)})"
        if isinstance(value, ModuleType):
            return f"module({value.__name__})"
        if isinstance(value, np.dtype):
            return f"dtype({_dtype_key(value)})"
        if isinstance(value, (llvm_ir.Type, llvm_ir.Value)):
            # LLVM IR a code generator builds on: its text is what it emits.
            return f"llvm({value})"
        try:
            return _canon_value(value, set())
        except (_Unfingerprintable, RecursionError):
            pass
        if isinstance(value, WithContext):
            # numba compiles a ``with`` block over one of its contexts, ``objmode`` say, from the block's own code.
            return f"withcontext({type(value).__module__}.{type(value).__qualname__};{_package_version('numba')})"
        if self.registered(value):
            # numba types it through a registered template: numpy's array functions, which are not Python functions.
            return self.overloaded(value)
        raise _Unkeyable(f"{where} is of type {type(value).__name__}, which has no stable form")


def _walk(py_func, keyed, codegen) -> _Walk:
    kind, compiler = keyed
    walk = _Walk()
    try:
        walk.parts["numbox"] = numbox.__version__
        walk.parts["settings"] = _settings_text(codegen)
        walk.parts["flags"] = walk.flags(kind, compiler)
        walk.function(py_func)
    except _Unkeyable:
        raise
    except RecursionError:
        raise _Unkeyable("the functions it calls nest too deeply to walk") from None
    except Exception as exc:
        raise _Unkeyable(f"reading it for the key failed ({exc!r})") from exc
    return walk


def _digest(text) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def _key_parts(py_func, keyed, codegen) -> tuple:
    return tuple(sorted((name, _digest(text)) for name, text in _walk(py_func, keyed, codegen).parts.items()))


def explain_cache_key(func) -> dict:
    """The parts numbox adds to ``func``'s cache key, as they stand for its next compile in this process.

    ``func`` is a dispatcher or ``@cfunc`` whose cache numbox keys: a function numbox compiled, or one decorated with
    :func:`keyed_njit` or :func:`keyed_cfunc` with ``cache=True``. The result maps ``"numbox"`` to numbox's version,
    ``"flags"`` to the flags as the key holds them, ``"settings"`` to each keyed setting's pair of values (when numba
    built its code generator, and now), and every other part, the code of each function the walk reaches and each
    global, closure cell and module attribute they read, to its digest. Two calls that disagree on a part are two
    different keys. ``ValueError`` names the value with no stable form where the function compiles uncached.
    """
    cache = getattr(func, "_cache", None)
    keyed = getattr(cache, "_numbox_keyed", None)
    if keyed is None:
        raise ValueError(
            f"{func!r} has no cache numbox keys: it is cached under numba's key alone, or not cached; decorate it "
            "with keyed_njit or keyed_cfunc and cache=True"
        )
    context = registry.cpu_target.target_context
    context.refresh()
    codegen = context.codegen()
    try:
        walk = _walk(cache._py_func, keyed, codegen)
    except _Unkeyable as exc:
        raise ValueError(f"{func!r} compiles without a cache: {exc}") from None
    built, now = _settings_at_build(codegen), _current_settings()
    explained = {
        "numbox": walk.parts["numbox"],
        "flags": walk.parts["flags"],
        "settings": {env: (at_build, value) for (env, at_build), (_, value) in zip(built, now)},
    }
    for name, text in sorted(walk.parts.items()):
        if name not in explained:
            explained[name] = _digest(text)
    return explained


_keying = threading.local()


@contextlib.contextmanager
def _keyed(func):
    """Mark the cache numba sets up for ``func`` while this is open, so numbox keys it."""
    stack = getattr(_keying, "funcs", None)
    if stack is None:
        stack = _keying.funcs = []
    stack.append(func)
    try:
        yield
    finally:
        stack.pop()


def _keys(py_func) -> bool:
    """Whether numbox keys the cache of ``py_func``: numbox's own code, or a function being decorated by numbox."""
    module = getattr(py_func, "__module__", None) or ""
    if module == "numbox" or module.startswith("numbox."):
        return True
    return any(f is py_func for f in getattr(_keying, "funcs", ()))


def keyed_njit(*args, **kwargs):
    """``numba.njit``, taking its arguments unchanged, with numbox's cache key on the function it decorates.

    With ``cache=True`` the function is cached under numba's key plus numbox's part of it: see the module documentation
    for what that holds. The key is set up as numba sets the cache up, at decoration, so it applies to an eager compile
    (a signature given) as to a lazy one. Without ``cache=True`` this is ``numba.njit``.
    """
    if args and isinstance(args[0], FunctionType):
        with _keyed(args[0]):
            return njit(*args, **kwargs)
    decorator = njit(*args, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


def keyed_cfunc(sig, **kwargs):
    """``numba.cfunc``, taking its arguments unchanged, with numbox's cache key on the function it decorates.

    With ``cache=True`` the C callback is cached under numba's key plus numbox's part of it, which tells it apart from
    an ``njit`` of the same function as well. Without ``cache=True`` this is ``numba.cfunc``.
    """
    decorator = cfunc(sig, **kwargs)

    def wrap(func):
        with _keyed(func):
            return decorator(func)
    return wrap


_uncached_reported = set()
_churn_reported = set()


def _qualname(cache) -> str:
    py_func = getattr(cache, "_py_func", None)
    return getattr(py_func, "__qualname__", None) or getattr(cache, "_name", "?")


def _warn_uncached(cache, exc):
    report = (getattr(cache, "_name", None), str(exc))
    if report in _uncached_reported:
        return
    _uncached_reported.add(report)
    warnings.warn(
        f"numbox: {_qualname(cache)} is compiled without an on-disk cache: {exc}. {exc.why}",
        UncachedWarning, stacklevel=4,
    )


def _check_churn(cache):
    """Warn, once per function and process, where its index has outgrown :data:`CHURN_THRESHOLD`.

    The parts that differ between the entries are named, most values first: a global that takes a new value every
    run, a timestamp or a random seed, saves a new entry every run. Nothing is removed. Different signatures are
    entries of their own too, and draw no warning while every part agrees.
    """
    try:
        name = getattr(cache, "_name", None)
        if name in _churn_reported:
            return
        index = cache._cache_file._load_index()
        if len(index) <= CHURN_THRESHOLD:
            return
        values = {}
        for key in index:
            parts = key[-1] if isinstance(key, tuple) and len(key) == 4 else None
            if not isinstance(parts, tuple):
                continue
            for part, digest in parts:
                values.setdefault(part, set()).add(digest)
        varying = sorted(
            ((len(digests), part) for part, digests in values.items()
             if len(digests) > 1 and part not in ("numbox", "flags", "settings")),
            key=lambda item: (-item[0], item[1]),
        )
        if not varying:
            return
        _churn_reported.add(name)
        named = ", ".join(f"{part} ({count} values)" for count, part in varying)
        warnings.warn(
            f"numbox: the cache of {_qualname(cache)} holds {len(index)} entries, and these parts of its key differ "
            f"between them: {named}. Each new value compiles and saves a new entry; a value that changes every run "
            "is better passed as an argument.",
            CacheChurnWarning, stacklevel=4,
        )
    except Exception:
        return


def _wrap_index_key(original):
    @functools.wraps(original)
    def _index_key(self, sig, codegen):
        key = original(self, sig, codegen)
        keyed = getattr(self, "_numbox_keyed", None)
        if keyed is None:
            return key
        return key + (_key_parts(self._py_func, keyed, codegen),)
    return _index_key


def _wrap_load_overload(original):
    @functools.wraps(original)
    def load_overload(self, sig, target_context):
        try:
            return original(self, sig, target_context)
        except _Unkeyable as exc:
            _warn_uncached(self, exc)
            return None
    return load_overload


def _wrap_save_overload(original):
    @functools.wraps(original)
    def save_overload(self, sig, data):
        try:
            original(self, sig, data)
        except _Unkeyable as exc:
            _warn_uncached(self, exc)
            return
        if getattr(self, "_numbox_keyed", None) is not None:
            _check_churn(self)
    return save_overload


def _wrap_enable_caching(original, kind, py_func_of):
    @functools.wraps(original)
    def enable_caching(self):
        original(self)
        if _keys(py_func_of(self)):
            self._cache._numbox_keyed = (kind, self._compiler)
    return enable_caching


def _wrap_codegen_init(original):
    @functools.wraps(original)
    def _init(self, llvm_module):
        self._numbox_settings_at_build = _current_settings()
        return original(self, llvm_module)
    return _init


def _install():
    """Wrap numba's cache, the two decorators' cache set-up and the code generator's set-up; once per process.

    Every lookup happens before the first assignment, so a numba that lacks one of them leaves no half-installed key.
    A code generator numba built before numbox was imported takes the settings as they are now.
    """
    if getattr(caching, "_numbox_cache_key", False):
        return
    wrapped = [
        (caching.Cache, "_index_key", _wrap_index_key(caching.Cache._index_key)),
        (caching.Cache, "load_overload", _wrap_load_overload(caching.Cache.load_overload)),
        (caching.Cache, "save_overload", _wrap_save_overload(caching.Cache.save_overload)),
        (Dispatcher, "enable_caching", _wrap_enable_caching(Dispatcher.enable_caching, "njit", lambda d: d.py_func)),
        (CFunc, "enable_caching", _wrap_enable_caching(CFunc.enable_caching, "cfunc", lambda c: c._pyfunc)),
        (numba_codegen.CPUCodegen, "_init", _wrap_codegen_init(numba_codegen.CPUCodegen._init)),
    ]
    built = vars(registry.cpu_target).get("_toplevel_target_context")
    caching._numbox_cache_key = True
    for owner, name, replacement in wrapped:
        setattr(owner, name, replacement)
    if built is not None:
        codegen = built.codegen()
        if getattr(codegen, "_numbox_settings_at_build", None) is None:
            codegen._numbox_settings_at_build = _current_settings()


try:
    _install()
except Exception as _exc:
    warnings.warn(
        f"numbox: could not add its part to numba's cache key ({_exc!r}); what numbox compiles is cached under "
        "numba's key alone",
        RuntimeWarning,
    )
