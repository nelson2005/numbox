numbox.core.proxy
=================

Overview
++++++++

Implementation of :func:`numbox.core.proxy.proxy` decorator that swaps definition of a jit-compiled
function in-place for a declaration (while delegating the actual implementation
to a different function that is only accessible indirectly). As a result, statically linking in libraries
corresponding to proxy-jitted functions called from other jitted functions will
only paste a declaration rather than the entire LLVM IR code.

The ``.as_func`` first-class value
++++++++++++++++++++++++++++++++++

Besides being callable, the dispatcher a ``@proxy`` decoration returns exposes
``.as_func``: a first-class function value for the main signature, to hand to a
jitted function that takes the binding as a ``FunctionType`` argument, or to
reference from jitted scope as a constant.

From numba 0.61 onward ``.as_func`` is a
:class:`~numbox.utils.derive_wap.DeriveWAP` typed as
:class:`~numbox.utils.derive_wap.DeriveFunctionType`, so the ``jit_addr`` slot of
its data model carries the numba calling convention entry point and an exception
raised inside the proxied body propagates out of a first-class call instead of
being discarded. numba 0.60 has no such slot, so ``.as_func`` is a plain
``CompileResultWAP`` there and the exception is still discarded. ``DeriveWAP``
subclasses ``CompileResultWAP``, so ``isinstance`` passes on either version and
only ``type(...) is CompileResultWAP`` tells the two apart.

Only one of the binding's two handles propagates. **Calling the dispatcher itself
still discards**, on every numba version: that call reaches the proxied body
through its C-convention cfunc wrapper, which does not unwind, so the exception
surfaces only as an unraisable on stderr and the call returns a zero-filled
value. Handing ``.as_func`` from Python into a jitted parameter *declared* as a
plain ``FunctionType`` discards as well, because the value degrades to the C
convention on the way in; ``DeriveFunctionType.can_convert_to`` permits that
conversion deliberately, so such a call site goes on compiling unchanged. An
inferred-signature ``@njit`` argument keeps the numbox type and propagates.

``.as_func`` inherits the mixed-container limit of any derive value: a tuple
holding it alongside a plain ``CompileResultWAP`` no longer unifies, failing with
a message-less ``AssertionError`` from
``numba.core.utils.unified_function_type``. See :doc:`numbox.core.work` for that
limit and for why making the two types compare equal is not available as a fix.

Cache-anchor mechanism
++++++++++++++++++++++

The ``@proxy`` decorator generates a thin wrapper function via ``exec()``.
For numba to cache that wrapper across processes, the wrapper's bytecode
needs a ``co_filename`` and ``co_firstlineno`` that point at real Python
source — both because numba's cache stamp uses ``(st_mtime, st_size)`` of
``co_filename`` for invalidation, and because
``inspect.getsourcelines(wrapper)`` gets called during numba's annotation
pipeline.

Anchoring at the user's file
----------------------------

The wrapper anchors at ``inspect.getfile(func)`` — the user's ``.py``
file where the ``@proxy`` decoration sits. Blank lines are prepended to
the generated wrapper source so the wrapper's ``@njit`` decorator
(which is what Python records as ``co_firstlineno`` for a decorated
function) lands at ``func.__code__.co_firstlineno`` — i.e. exactly the
line of the user's ``@proxy`` decorator. ``inspect.findsource(wrapper)``
then matches that ``@proxy`` line on its first check via
``r'^(\s*@)'`` — no backward scan needed, tokenization proceeds from
real, syntactically valid Python.

The hazard this avoids
----------------------

``inspect.findsource`` searches backward from ``co_firstlineno`` for any
line matching its pattern, *including* lines inside docstrings that
happen to start with ``@``. A docstring mentioning
``@njit(parallel=True) workers`` indented four spaces would be matched
as if it were a real decorator, and the C tokenizer would then read
``worker's`` (the apostrophe) as an unterminated string literal and
raise ``TokenError``. Related to CPython issue
`#122981 <https://github.com/python/cpython/issues/122981>`_.

Placing the wrapper's ``co_firstlineno`` directly at the user's
``@proxy`` line means ``findsource`` matches without scanning, and the
docstring contents are never re-tokenized.

Cache invalidation
------------------

For a file-backed cached ``@njit``, numba's source stamp is
``(os.stat(co_filename).st_mtime, st_size)`` — see
``numba.core.caching._SourceFileBackedLocatorMixin.get_source_stamp``.
Any edit to the user's file (the file containing the ``@proxy``
decoration) invalidates the wrapper's cache. Edits to ``proxy.py``'s
wrapper template itself — without a corresponding user-file edit — do
*not* invalidate the cache (the user file's mtime is unchanged); treat
wrapper-template changes as developer-managed. Clear the affected
entries — the ``.nbc`` / ``.nbi`` files in the ``__pycache__`` beside
each binding's source, or under ``NUMBA_CACHE_DIR`` — when shipping a
template change to numbox.

A second staleness shape sits outside the wrapper's anchor entirely. A
``cache=True`` caller that reaches a binding's ``.as_func`` as a compile-time
constant is newly cacheable, and none of the stamps above notice when the proxied
body changes underneath it. Constant-lowering a plain ``CompileResultWAP`` bakes
the entry point into a dynamic global, and numba refuses to cache a function
carrying one, so such a caller used to recompile in every process and emit
``NumbaWarning: Cannot cache compiled function ... as it uses dynamic globals``.
``lower_constant_derive_function_type`` (``numbox/utils/derive_wap.py``) instead
declares the ``jit_addr`` slot symbolically with ``context.declare_function`` and
links the proxied body's compile result in with ``add_linking_library``. The call
site then uses only that slot, so the ``c_addr`` and ``py_addr`` dynamic globals
are dead and get eliminated before numba scans the final module, and the caller
caches. This holds from numba 0.61 onward; on 0.60 ``.as_func`` is a plain
``CompileResultWAP`` and the caller stays uncacheable as before.

What such a caller caches is the proxied body's machine code, linked in. Its
cached binary binds the body as it stood at compile time, while numba's freshness
stamp watches only the caller's own source file. **Editing a proxied body and
rerunning without clearing the numba cache serves the old numbers.** The two
handles can disagree within one process: a dispatcher-path caller of the same
binding recompiles and returns the new value while the const-referencing caller
returns the old one.

The stale-alias guard described below cannot cover this, by construction. It
scans a cached object's undefined symbols for the ``numbox_pxy_`` prefix, and a
constant reference emits no such symbol, because the body is linked into the
object rather than referenced through the alias. The single
:class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` such a run emits names
the dispatcher-path caller that healed correctly, never the const-referencing
caller serving the stale value, and the run after that is silent. Reverting to a
numbox that mints a plain ``CompileResultWAP`` is not a remedy either: numba's
index key is callee-blind, so it loads the entry already on disk. Clearing the
cache is the remedy.

Both halves are pinned in ``test/core/test_proxy_cache_stale.py``, by
``test_a_const_reference_caller_becomes_cacheable`` and
``test_a_const_reference_caller_serves_a_stale_body_after_an_edit``.

Alias content-addressing and cross-file callers
------------------------------------------------

``@proxy`` references the proxied body through a deterministic symbol alias
(``numbox_pxy_<name>_<hash>``) registered per process via
``llvmlite.binding.add_symbol``. The hash folds the body's content fingerprint
alongside ``module``, ``qualname`` and the signature, so two different bodies
that share one identity — factory-made same-qualname closures, an in-process
redefinition, or ``fork()`` twins on a shared cache — get distinct aliases
instead of colliding on one (a collision let the last ``add_symbol`` win and
silently rebound callers to the wrong body).

Because the alias encodes the body, changing a proxied binding's **signature or
body renames its alias**. numba's cache key for a *caller* is callee-blind, so a
``cache=True`` caller in another file cache-hits unchanged after such a change
and references the old alias, which the new process never registers. Left
unhandled this is a hard crash rather than a miss: RuntimeDyld resolves an
object's externals in one batch, so the single missing name zeroes *every*
relocation in the object and the process dies inside the argument-unpacking
wrapper — a bare segfault with an empty stderr, or, once any later cached object
is loaded, an ``LLVM ERROR: Symbol not found`` abort.

**numbox heals this on load; no manual cache clearing is needed.** Importing any
``@proxy`` binding imports numbox, which installs a guard around numba's cache
``rebuild``. Before a cached object reaches the execution engine its undefined
``numbox_pxy_*`` symbols are checked against the registered aliases; an object
referencing one this process never registered is discarded and recompiled in
place, emitting a :class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` that
names the retired alias. ``@njit``, ``@vectorize``, ``@guvectorize`` and
``@cfunc`` callers all heal.

Set ``NUMBOX_PROXY_CACHE_STRICT`` (truthy: anything other than unset / ``0`` /
``false`` / ``no`` / ``off``, case-insensitive) to make the guard fail loud
instead of healing: a stale alias then raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` *before* the discard,
leaving the stale entry on disk to inspect, and a payload the guard cannot read
raises :class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheError` rather than
loading unchecked.

**First run after an upgrade.** A numbox release that changes the alias
fingerprint renames every shipped alias at once, so the first process after the
upgrade heals every warm caller of a numbox binding — a one-time burst of
recompiles and warnings, after which the cache is warm again. To clear it by hand
instead, the entries are the ``.nbc`` / ``.nbi`` files in the ``__pycache__``
directory beside each caller's own source (or under ``NUMBA_CACHE_DIR`` if set);
``~/.cache/numba`` holds only callers numba cannot anchor to a source file.

The one variant that leaves the alias unchanged — a ``proxy_if_available``
binding present when the caller was cached but absent on reload — would otherwise
resolve to a diagnostic trap (a cfunc registered under the alias whose
``RuntimeError`` numba swallows at the C boundary, returning zero); the guard
treats such an alias as stale too, so the caller reaches the same clean typing
error a cold cache gives.

Multi-decorator support
-----------------------

When a user stacks decorators above ``@proxy(sig)``,
``func.__code__.co_firstlineno`` is the topmost decorator line (Python
records a decorated function's first line as its outermost decorator).
The anchor lands the wrapper at that topmost decorator.
``findsource`` matches it directly because every decorator line begins
with ``@``. Verified for single-, double-, and triple-stack outer
decorators.

``@proxy`` itself must be the innermost decorator (closest to ``def``).
A wrapping decorator between ``@proxy`` and ``def`` would hand
``@proxy`` a wrapped function whose ``__code__`` lives in the wrapping
decorator's source file, and would also break numba's ability to
JIT-compile through the intermediate Python wrapper.

Modules
+++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
