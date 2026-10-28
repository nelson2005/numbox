numbox.core.proxy
=================

Overview
++++++++

Implementation of :func:`numbox.core.proxy.proxy` decorator that swaps definition of a jit-compiled
function in-place for a declaration (while delegating the actual implementation
to a different function that is only accessible indirectly). As a result, statically linking in libraries
corresponding to proxy-jitted functions called from other jitted functions will
only paste a declaration rather than the entire LLVM IR code. One route is exempt: a jitted caller
that reaches a binding's ``.as_func`` as a compile-time constant links in the body's code rather than
a declaration, with the caching consequence described under `Cache invalidation`_ below.

The ``.as_func`` first-class value
++++++++++++++++++++++++++++++++++

Besides being callable, the dispatcher that a ``@proxy`` decoration returns exposes ``.as_func``: a first-class
function value for the main signature. It can be passed to a jitted function that takes the binding as a
``FunctionType`` argument, or referenced from jitted scope as a constant.

Passing it as an argument is what keeps the *receiving* function cacheable. If a ``@njit(cache=True)`` function is
passed the dispatcher itself, that parameter is typed as ``type(CPUDispatcher(...))``, whose name carries the
dispatcher's address, so the cache index key differs in every process. The receiving function then never matches
the entry an earlier run wrote: it recompiles on every run and adds another overload to the index.
If it is passed ``.as_func``, the parameter types as a function type instead,
``DeriveFunctionType[float64(float64)]``, which is structural and identical in every process, so the receiving
function is compiled once and loaded from the cache in every later process.

``.as_func`` is a :class:`~numbox.utils.derive_wap.DeriveWAP` typed as
:class:`~numbox.utils.derive_wap.DeriveFunctionType`, so the ``jit_addr`` slot of its data model carries the numba
calling convention entry point, and an exception raised inside the proxied body propagates out of a first-class
call instead of being discarded. ``DeriveWAP`` subclasses ``CompileResultWAP``, so an ``isinstance`` check against
``CompileResultWAP`` passes for both, and only ``type(...) is CompileResultWAP`` distinguishes the two.

Only one of the binding's two handles propagates an exception.
**Calling the dispatcher itself still discards the exception**, on every numba version.
That call reaches the proxied body through its C-convention cfunc wrapper, which does not unwind, so the exception is
reported only as an unraisable on stderr and the call returns a zero-filled value.

Passing ``.as_func`` from Python into a jitted parameter *declared* as a plain ``FunctionType`` discards the
exception as well, because the value is converted to the C convention at the boundary.
``DeriveFunctionType.can_convert_to`` permits that conversion deliberately, so such a call site continues to compile
unchanged.
An inferred-signature ``@njit`` argument keeps the numbox type and propagates.

Below numba 0.68.0, ``.as_func`` inherits the mixed-container limit of any derive value: a tuple holding it
alongside a plain ``CompileResultWAP`` does not unify, failing with a message-less ``AssertionError`` from
``numba.core.utils.unified_function_type``; numba 0.68.0 unifies it.
See :doc:`numbox.core.work` for that limit and for why making the two types
compare equal is not available as a fix.

Referencing ``.as_func`` as a constant has a caching caveat that passing it as a function-type argument does not
have. Read `Cache invalidation`_ below before capturing one into a module-level global.

Cache-anchor mechanism
++++++++++++++++++++++

The ``@proxy`` decorator generates a thin wrapper function with ``exec()``.
numba caches that wrapper across processes only if the wrapper's bytecode has a ``co_filename`` and a
``co_firstlineno`` that point at real Python source. Two things need them: numba's cache stamp uses the ``(st_mtime,
st_size)`` of ``co_filename`` for invalidation, and numba's annotation pipeline calls
``inspect.getsourcelines(wrapper)``.

Anchoring at the user's file
----------------------------

The wrapper anchors at ``inspect.getfile(func)``, the user's ``.py`` file that holds the ``@proxy`` decoration.
Blank lines are prepended to the generated wrapper source, so that the wrapper's ``@njit`` decorator, which Python
records as ``co_firstlineno`` for a decorated function, is placed at ``func.__code__.co_firstlineno``: exactly the
line of the user's ``@proxy`` decorator. ``inspect.findsource(wrapper)`` then matches that ``@proxy`` line on its
first check with ``r'^(\s*@)'``. No backward scan is needed, and tokenization proceeds from real, syntactically
valid Python.

The hazard this avoids
----------------------

``inspect.findsource`` searches backward from ``co_firstlineno`` for any line that matches its pattern,
*including* lines inside docstrings that happen to start with ``@``. A docstring mentioning
``@njit(parallel=True) workers``, indented four spaces, would be matched as if it were a decorator, and the C
tokenizer would then read ``worker's`` (the apostrophe) as an unterminated string literal and raise
``TokenError``. Related to CPython issue
`#122981 <https://github.com/python/cpython/issues/122981>`_.

Placing the wrapper's ``co_firstlineno`` directly at the user's ``@proxy`` line means that ``findsource`` matches
without scanning, and the docstring contents are never tokenized again.

Cache invalidation
------------------

For a file-backed cached ``@njit``, numba's source stamp is ``(os.stat(co_filename).st_mtime, st_size)``; see
``numba.core.caching._SourceFileBackedLocatorMixin.get_source_stamp``. Any edit to the user's file, the file that
holds the ``@proxy`` decoration, invalidates the wrapper's cache. An edit to the wrapper template in ``proxy.py``
itself, without an edit to the user's file, does *not* invalidate the cache, because the user file's mtime is
unchanged; a wrapper-template change is managed by the developer.
When a template change goes into numbox, clear the affected entries: the ``.nbc`` and ``.nbi`` files in the
``__pycache__`` beside each binding's source, or under ``NUMBA_CACHE_DIR``.

A second kind of staleness lies outside the wrapper's anchor entirely. A ``cache=True`` caller that reaches a
binding's ``.as_func`` as a compile-time constant is newly cacheable, and none of the stamps above detects a change
to the proxied body.
Constant-lowering a plain ``CompileResultWAP`` embeds the entry point in a dynamic global, and numba refuses to cache
a function that carries one. Such a caller used to recompile in every process and emit ``NumbaWarning: Cannot cache
compiled function ... as it uses dynamic globals``.
``lower_constant_derive_function_type`` (``numbox/utils/derive_wap.py``) instead declares the ``jit_addr`` slot
symbolically with ``context.declare_function`` and links the proxied body's compile result in with
``add_linking_library``. The call site then uses only that slot, so the ``c_addr`` and ``py_addr`` dynamic globals
are dead and are eliminated before numba scans the final module, and the caller caches.

What such a caller caches is the proxied body's machine code, linked in. The inlining on this path is not the
``inline='always'`` that the decorator puts on the generated wrapper.
That option is a numba-IR inline of the wrapper's single ``@intrinsic`` call; it applies only where a jitted caller
calls the dispatcher, and its result is the alias declaration on which the static-linking avoidance is built.
A constant reference never reaches
the wrapper. ``.as_func`` carries the *body's* compile result, ``lower_constant_derive_function_type`` links that
library into the caller, and LLVM then inlines the body itself for an ordinary body size, so on this one path the
static-linking avoidance does not apply. The cached binary binds the body as it was at compile time, while numba's
freshness stamp checks only the caller's own source file. **This is the constant-reference route alone: a caller
that reaches the same binding through its dispatcher is repaired on load by the stale-alias guard described below,
and needs no manual clearing.
Clearing the numba cache after an edit to the proxied body is mandatory, not merely advisable, for a caller that
holds the body as a compile-time constant.**

Which body such a caller runs is not single-valued. A body small enough to inline leaves no separate definition
behind, and the caller returns the old values in every later process. A larger body is embedded in the caller's
object as a weak ``linkonce_odr`` definition under a mangled name that carries numba's per-process ``v<uid>`` abi
tag, which numba's cache key does not cover. Which body that caller reaches then depends on the name the proxied
body carries in the loading process. That name is assigned by whichever process last recompiled the body and is
stored in the body's own cache entry, so it need not track the loading process's own counter.
The caller runs its embedded copy where that name differs from the embedded name, while the dispatcher path runs the
new body.
Where that name matches, the caller binds to the definition the already-loaded body registered, and both handles
return the new value while the caller's cached object still holds only the old code.
The first process to recompile the body after an edit determines which of the two every later process receives.
Both outcomes produce no warning, so the
two handles disagreeing within one process is one symptom and not a dependable one: in the non-inlined case they
can agree while the cached caller runs a body it never embedded.

The stale-alias guard described below cannot cover this, by construction. It scans a cached object's undefined
symbols for the ``numbox_pxy_`` prefix, and a constant reference emits no such symbol, because the body is linked
into the object rather than referenced through the alias. The single
:class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` such a run emits names the dispatcher-path caller that
was repaired, never the const-referencing caller that returns the stale value, and the run after that emits no
warning. Reverting to a numbox that produces a plain ``CompileResultWAP`` is not a remedy either: numba's index key
does not distinguish callees, so it loads the entry already on disk. Clearing the cache is the remedy.

The gap covers more than a body edit. A ``proxy_if_available`` binding that was present when the caller was
cached but is absent in a later process is not caught either. The const-referencing caller hits the cache and
calls into a binding that is not present.
Where the body needs no external symbol, the caller returns a value computed by the absent binding; where the body
does need one, the process terminates with a bare segfault and an empty stderr.
A cold cache in the same process raises a typing error instead, so the warm run and the cold run disagree.
``NUMBOX_PROXY_CACHE_STRICT`` does not catch it either, for the same reason the guard does not: there is no alias
to check.
The deterministic case, a body that needs no external symbol, so that the warm run returns the absent binding's own
value instead of segfaulting, is covered in ``test/core/test_proxy_cache_stale.py`` by
``test_a_const_reference_caller_calls_a_binding_that_disappeared``.

Guarding the use with ``hasattr(binding, "as_func")`` protects in two cases only. The first is a jitted caller whose
*definition* is inside that ``if``, so that the absent process never defines it.
The second is ``.as_func`` passed as a function-type argument rather than referenced as a constant, because the
address is then unboxed per call and the absent process uses the fallback.
The case that fails is ``binding.as_func if hasattr(...) else fallback``
assigned to a module global while the ``cache=True`` caller is defined unconditionally, since numba loads the cache
entry before it types anything and the ``else`` branch is never consulted. Capturing ``.as_func`` with no guard at
all is safe: the attribute is missing in the absent process, so the import fails with an error and the cache is
never reached.

The body-edit case is covered in ``test/core/test_proxy_cache_stale.py``, by
``test_a_const_reference_caller_becomes_cacheable`` and
``test_a_const_reference_caller_serves_a_stale_body_after_an_edit``.

Alias content-addressing and cross-file callers
------------------------------------------------

``@proxy`` references the proxied body through a deterministic symbol alias (``numbox_pxy_<name>_<hash>``),
registered per process with ``llvmlite.binding.add_symbol``.
The hash folds the body's content fingerprint together with ``module``, ``qualname`` and the signature.
Two different bodies that share one identity therefore receive distinct aliases instead of colliding on one: closures
of the same qualname made by a factory, an in-process redefinition,
or two bodies in processes related by ``fork()`` on a shared cache.
A collision made the last ``add_symbol`` registration prevail and rebound callers to the wrong body with no warning.

Separating them requires the fingerprint to include the captured value that differs, and what it can include is
bounded by what has an identity that means the same thing in the next process. A
``numba.core.types.ExternalFunction`` and a ctypes pointer reached by attribute access on a loaded library both
have such an identity, and both are folded. A ctypes pointer built from a raw address does not: two of them differ
only by an address that ASLR moves, so folding it would gain discrimination at the cost of the process stability
the alias exists for.

Bodies in that last class are therefore allowed to collide, and the collision is detected rather than hidden.
The second body is published under a process-local name of its own, so both bodies stay callable and correct in this
process.
The shared name is retired into the absent-alias set, so a warm caller of either one is discarded and
recompiled rather than bound to whichever body holds the name at the time.
Both lose cross-process caching, which
is the cost of a name that does not identify what it names, and an
:class:`~numbox.core.proxy.proxy.AliasCollisionWarning` reports it.

Reaching an alias twice is usually not a collision.
The same body is registered again whenever a module is reloaded or a factory is called twice with equal arguments.
Both registrations name the same code, so the alias is re-pointed at the newly compiled address and nothing is
retired.

The two cases are distinguished by the values the fingerprint could not canonicalize.
Two bodies that fingerprint alike are the same body except in those values, so comparing them determines whether one
body was compiled twice or two bodies were given one name.
That comparison is required only within the current process, which is why it may read an address that the fingerprint
may not.
The address alone is not the body, though: one symbol reached through two ctypes prototypes is two calls that numba
types and lowers differently. The prototype's shape is therefore compared beside the address, and only the two
together establish that one body was compiled twice.

A name already retired by a collision stays retired through such a re-registration.
Re-pointing it does not make it identify a body again, because another process may still build the two bodies in the
opposite order, so no alias is removed from the absent-alias set.

Retiring the two names does not contain the collision fully, because a third body can capture one of them.
A ``@proxy`` body that merely calls a colliding binding closes over its dispatcher.
Folding that dispatcher by the name it currently holds would make the derived body's own alias a function of the
order in which this process built the pair. Two such derived bodies would then exchange names between one process and
the next.

Neither derived name would be retired, since each is newly created and resolves normally.
Unlike the pair it wraps, a derived body would therefore keep the cross-process cache that has just become wrong, and
the collision would propagate one composition step.
The walker therefore declines to fold a retired alias. A body that captured one falls through to the ordinary
walk, where a dispatcher has no canonical form, so it meets the same comparison every un-canonicalizable value
meets and its own alias is retired in its turn.

One case remains open.
A derived body fingerprinted *before* the collision that retires its dependency folded a name that still identified a
body, and it is already published under a name of its own that is never revisited.
Closing the case would mean recording which aliases each fingerprint consumed and retiring dependents transitively,
so what is documented here is the bound. Containment reaches a derived body built after the collision, and the
warning's promise that callers of both recompile in every process holds for everything built from that point on.

Because the alias encodes the body, changing a proxied binding's **signature or body renames its alias**. numba's
cache key for a *caller* does not distinguish callees, so a ``cache=True`` caller in another file hits the cache
unchanged after such a change and references the old alias, which the new process never registers.
Left unhandled, this is a hard crash rather than a miss. RuntimeDyld resolves an object's externals in one batch, so
the single missing name zeroes *every* relocation in the object, and the process terminates inside the
argument-unpacking wrapper. The result is a bare segfault with an empty stderr, or, once any later cached object is
loaded, an ``LLVM ERROR: Symbol not found`` abort.

**numbox repairs this on load; no manual cache clearing is needed.** Importing any ``@proxy`` binding imports
numbox, which installs a guard around numba's cache ``rebuild``.
The guard checks a cached object's undefined ``numbox_pxy_*`` symbols against the registered aliases before the
object reaches the execution engine. An object that references an alias this process never registered is discarded
and recompiled in place, with a :class:`~numbox.core.proxy.proxy.StaleProxyCacheWarning` that names the retired
alias.
``@njit``, ``@vectorize``,
``@guvectorize`` and ``@cfunc`` callers are all repaired.

Set ``NUMBOX_PROXY_CACHE_STRICT`` (truthy: anything other than unset, empty or whitespace, ``0``, ``false``, ``no``
or ``off``, case-insensitive) to make the guard raise instead of repairing.
A stale alias then raises
:class:`~numbox.core.proxy.proxy.StaleProxyCacheError` *before* the discard, which leaves the stale entry on disk for
inspection.
Under the same setting, a payload the guard cannot read raises
:class:`~numbox.core.proxy.proxy.UnvalidatedProxyCacheError` rather than loading unchecked.

**First run after an upgrade.**
A numbox release that changes the alias fingerprint renames every shipped alias at once, so the first process after
the upgrade repairs every warm caller that reaches a numbox binding through its alias. That is a one-time batch of
recompiles and warnings, after which the cache is warm again.
A caller that reached ``.as_func`` as a compile-time constant is not among them: it carries no alias, so it goes on
running the pre-upgrade body on every run until its own cache entry is cleared.
To clear it manually instead, delete the ``.nbc`` and ``.nbi`` files in the ``__pycache__`` directory beside each
caller's own source, or under ``NUMBA_CACHE_DIR`` if set.
They are in numba's user cache directory (see :ref:`where_the_cache_lands`) for a caller whose ``__pycache__`` numba
cannot write.

The one variant that leaves the alias unchanged is a ``proxy_if_available`` binding present when the caller was
cached but absent on reload. It would otherwise resolve to a diagnostic trap: a cfunc registered under the alias
whose ``RuntimeError`` numba discards at the C boundary, returning zero.
The guard treats such an alias as stale too, so a caller that reaches the binding through its alias gets the same
typing error a cold cache gives.
A
caller that reached ``.as_func`` as a compile-time constant emits no alias and is not covered; see
`Cache invalidation`_ above for what it gets instead.

Multi-decorator support
-----------------------

When a user stacks decorators above ``@proxy(sig)``, ``func.__code__.co_firstlineno`` is the topmost decorator
line, because Python records a decorated function's first line as its outermost decorator. The anchor places the
wrapper at that topmost decorator. ``findsource`` matches it directly, because every decorator line begins with
``@``. This is verified for one, two and three outer decorators.

``@proxy`` itself must be the innermost decorator, the one closest to ``def``.
A wrapping decorator between ``@proxy`` and ``def`` would pass ``@proxy`` a wrapped function whose ``__code__``
belongs to the wrapping decorator's source file, and would also prevent numba from JIT-compiling through the
intermediate Python wrapper.

Modules
+++++++

numbox.core.proxy.proxy
-----------------------

.. automodule:: numbox.core.proxy.proxy
   :members:
   :show-inheritance:
   :undoc-members:
