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
inferred-signature ``@njit`` argument keeps the numbox type and propagates, from
numba 0.61 onward; on 0.60 there is no numbox type to keep and it discards along
with the rest.

From numba 0.61 onward ``.as_func`` inherits the mixed-container limit of any
derive value: a tuple holding it alongside a plain ``CompileResultWAP`` no longer
unifies, failing with a message-less ``AssertionError`` from
``numba.core.utils.unified_function_type``. See :doc:`numbox.core.work` for that
limit and for why making the two types compare equal is not available as a fix.

Referencing ``.as_func`` as a constant makes a ``cache=True`` caller cacheable,
which passing it as a function-type argument does not; `Cache invalidation`_ below
covers what that means for a caller capturing one into a module-level global.

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

A second caching shape sits outside the wrapper's anchor entirely. A
``cache=True`` caller that reaches a binding's ``.as_func`` as a compile-time
constant is newly cacheable, and none of the stamps above notice when the proxied
body changes underneath it. Constant-lowering a plain ``CompileResultWAP`` bakes
the entry point into a dynamic global, and numba refuses to cache a function
carrying one, so such a caller used to recompile in every process and emit
``NumbaWarning: Cannot cache compiled function ... as it uses dynamic globals``.
``lower_constant_derive_function_type`` (``numbox/utils/derive_wap.py``) instead
fills the ``jit_addr`` slot with an *external* reference to a second numbox alias
(``numbox_pxy_cc_<name>_<hash>``), published over the derive's numba-callconv
entry point. The call site then uses only that slot, so the ``c_addr`` and
``py_addr`` dynamic globals are dead and get eliminated before numba scans the
final module, and the caller caches. This holds from numba 0.61 onward; on 0.60
``.as_func`` is a plain ``CompileResultWAP`` and the caller stays uncacheable as
before.

What such a caller caches is a *reference* to the proxied body, not its machine
code. The decorator's static-linking avoidance therefore holds on this path as it
does on the dispatcher path, and the caller's object comes out the same size
whether or not LLVM would have been willing to inline the body. That is what
makes the cached entry safe as well as small: the callconv alias folds the same
body fingerprint the cfunc alias does, so **editing a proxied body renames it, the
warm caller's object imports a name this process never registered, and the guard
described below discards that entry and recompiles it.** It is the same heal,
through the same guard, that a dispatcher-path caller gets, announced by the same
:class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning`.

The guard covers more than a body edit here too. A ``proxy_if_available`` binding
that was present when the caller was cached but is absent in a later process
registers no callconv alias in that process, so the warm entry is discarded and
the caller reaches the same clean typing error a cold cache gives, rather than
computing on the vanished binding's value or dying on a bare segfault with an
empty stderr. ``NUMBOX_PROXY_CACHE_STRICT`` sees it as well, and refuses the heal.

Guarding the use with ``hasattr(binding, "as_func")`` is therefore no longer what
decides whether such a caller is correct, but the shapes still differ in what they
report. Keeping the jitted caller's *definition* inside that ``if`` means the
absent process never defines it. Passing ``.as_func`` as a function-type argument
unboxes the address per call, so the absent process takes the fallback. Capturing
``.as_func`` with no guard at all fails loudly at import, because the attribute is
missing. ``binding.as_func if hasattr(...) else fallback`` assigned to a module
global, with the ``cache=True`` caller defined unconditionally, is the shape that
used to return the vanished binding's answer; it now recompiles against whatever
the ``else`` branch supplies.

One derive shape is deliberately left uncacheable: a foreign ``CompileResultWAP``
upgraded by :func:`~numbox.utils.derive_wap.rewrap_derive`. It carries a compile
result and nothing else, and a compile result numba restored from its own cache
has dropped the Python function, so there is no body to fingerprint and no alias
that would rename itself when the body changed. The lowering bakes the address
there, numba declines to cache the caller and says so on every run, and the caller
therefore always runs the current body. A derive minted by
:func:`~numbox.utils.highlevel.cres` is not in that position, because ``cres`` has
the function in hand, and it behaves exactly like a ``@proxy`` binding's
``.as_func``.

Both routes are pinned in ``test/core/test_proxy_cache_stale.py``, by
``test_a_const_reference_caller_becomes_cacheable``,
``test_a_const_reference_caller_heals_after_an_edit``,
``test_a_const_reference_caller_discards_a_binding_that_disappeared``,
``test_a_const_reference_caller_carries_no_copy_of_the_body``,
``test_a_cres_derive_reached_as_a_constant_heals_too`` and
``test_an_upgraded_foreign_wrapper_is_lowered_uncacheable``.
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
upgrade heals every warm caller that reaches a numbox binding through its alias —
a one-time burst of recompiles and warnings, after which the cache is warm again.
A caller that reached ``.as_func`` as a compile-time constant is among them: it
references a callconv alias built from the same fingerprint, so it is renamed and
healed in the same burst. To clear entries by hand instead, they are the
``.nbc`` / ``.nbi`` files in the ``__pycache__`` directory beside each
caller's own source (or under ``NUMBA_CACHE_DIR`` if set);
``~/.cache/numba`` holds only callers numba cannot anchor to a source file.

One entry the burst cannot reach is a const-reference caller cached by a numbox old
enough to have linked the body in rather than referenced it. Such an object names
no alias at all, so the guard passes it as it always did, and it goes on serving
the body it embedded until numba re-keys the caller for a reason of its own or the
entry is cleared by hand. Only entries written before the callconv alias existed
are in that position; everything written since is covered.

The one variant that leaves the alias unchanged — a ``proxy_if_available``
binding present when the caller was cached but absent on reload — would otherwise
resolve to a diagnostic trap (a cfunc registered under the alias whose
``RuntimeError`` numba swallows at the C boundary, returning zero); the guard
treats such an alias as stale too, so a caller that reaches the binding through
its alias gets the same clean typing error a cold cache gives. A caller that
reached ``.as_func`` as a compile-time constant is covered by the same rule, and
for a simpler reason: the absent path registers no callconv alias at all, so its
reference has nothing to resolve to.

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
