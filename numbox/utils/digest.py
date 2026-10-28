"""Content-addressed digest for cache keys and anchor identifiers.

``digest`` produces a short, stable hash that changes when the subject's ``repr``, the user functions, the resolved
``jit_options`` or the numba/numbox versions change. Plain Python functions are fingerprinted with the shared
closure/global-aware walker (``numbox.utils.fingerprint._fingerprint_function``). Two callbacks with identical source
but different captured closure-cell or referenced-global values therefore produce distinct digests, which a bare
code-object hash would not. A function closing over an un-canonicalizable value is fingerprinted by the best-effort
walker, which substitutes a placeholder for such a value. A callable with no code (a partial, a builtin, a callable
object) falls back to cloudpickle of the object, which captures its bound state. The SQLite UDAF registration anchors
are one consumer, and any content-addressed cache that combines a type or an identifier with user callbacks can reuse
the digest.
"""
import hashlib

from types import FunctionType

import numba
from numba.core.serialize import cloudpickle

import numbox
from numbox.core.configurations import jit_options
from numba.core.dispatcher import Dispatcher
from numba.np.ufunc.dufunc import DUFunc

from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function, _fingerprint_function_best_effort,
)


def digest(subject, fns):
    """Return a content hash that changes when any of its inputs changes.

    The inputs are ``subject`` (by ``repr``), the user functions, the resolved ``jit_options`` and the numba/numbox
    versions. The user functions are hashed through the shared fingerprint walker, which is sensitive to closures,
    globals and constants, with a cloudpickle fallback.
    """
    h = hashlib.sha256()
    h.update(repr(subject).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    # numbox.__version__ is "" upstream (the package version derives from it via pyproject's dynamic attr), so this
    # update currently has no effect. It is kept so that the digests change automatically if numbox ever sets a real
    # __version__.
    h.update((numbox.__version__ or "").encode("utf-8"))
    # Include the resolved numbox-wide jit_options, so that a change to NUMBOX_JIT_OPTIONS (such as turning the cache
    # off) re-keys the digest. numba's own cache also keys on the flags.
    h.update(repr(sorted(jit_options.items())).encode("utf-8"))
    for fn in fns:
        if isinstance(fn, (Dispatcher, DUFunc)):
            # Route jitted callables through the shared value canonicalizer, which includes targetoptions and honors
            # the content-addressed alias of a @proxy. The py_func shortcut below hashes only the Python body, so
            # without this, dispatchers over one body compiled with different jit flags, such as nogil, fastmath,
            # error_model or boundscheck, would all collide.
            try:
                h.update(_canon_value(fn, set()).encode("utf-8"))
                continue
            except (_Unfingerprintable, RecursionError):
                # The body cannot be canonicalized. Fold in the flags on their own, so that they still re-key the
                # digest, and fall through to the best-effort walker.
                h.update(repr(sorted((getattr(fn, "targetoptions", None) or {}).items())).encode("utf-8"))
        py = getattr(fn, "py_func", fn)
        if isinstance(py, FunctionType):
            try:
                h.update(_fingerprint_function(py, set()).encode("utf-8"))
            except (_Unfingerprintable, RecursionError):
                # An un-canonicalizable closure or global makes the strict walker raise. The best-effort walker still
                # captures the constants, closure cells, defaults and globals that it can. It substitutes opaque
                # placeholders for the rest and sorts sets, so its result is PYTHONHASHSEED-stable. A bare cloudpickle
                # of __code__ here would drop exactly the closure and default state that forced the fallback, and the
                # digest would depend on the iteration order of a set of str.
                h.update(_fingerprint_function_best_effort(py).encode("utf-8"))
            continue
        # A callable without code, such as a partial, a builtin or a callable object: cloudpickle the object to
        # capture its bound state. Only these callables take the cloudpickle path, so its uuid4 nondeterminism for
        # __main__ objects is confined to this case.
        h.update(cloudpickle.dumps(py))
    return h.hexdigest()[:16]
