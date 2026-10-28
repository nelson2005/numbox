import functools

from numba import njit, vectorize

from numbox.utils.digest import digest
from test.auxiliary_utils import collect_and_run_tests


def test_digest_is_deterministic():
    def cb(x):
        return x + 1
    assert digest("StateType", (cb,)) == digest("StateType", (cb,))


def test_digest_length_is_16():
    def cb(x):
        return x
    assert len(digest("StateType", (cb,))) == 16


def test_digest_varies_with_subject():
    def cb(x):
        return x
    assert digest("AType", (cb,)) != digest("BType", (cb,))


def test_digest_varies_with_function_literal():
    # The digest is sensitive to co_consts: bodies that differ only in a numeric literal must produce different
    # digests. The digest hashes cloudpickle of __code__, not the bare co_code, which is identical for these bodies.
    def cb_one(x):
        return x + 1

    def cb_two(x):
        return x + 2

    assert digest("StateType", (cb_one,)) != digest("StateType", (cb_two,))


def test_digest_varies_with_jit_options(monkeypatch):
    from numbox.utils import digest as digest_mod

    def cb(x):
        return x
    before = digest("StateType", (cb,))
    monkeypatch.setitem(digest_mod.jit_options, "cache", not digest_mod.jit_options.get("cache", True))
    assert digest("StateType", (cb,)) != before


def test_digest_accepts_codeless_callable():
    # A callable with no __code__ (a builtin, a functools.partial or a callable object) must be hashed through
    # cloudpickle of the object itself instead of raising.
    assert len(digest("StateType", (functools.partial(len),))) == 16


def test_digest_codeless_callable_captures_state():
    # The object fallback hashes the whole object, so partials with different arguments (state) yield different
    # digests. A hash of __call__.__code__ alone would not detect that difference.
    one = digest("StateType", (functools.partial(len, "a"),))
    two = digest("StateType", (functools.partial(len, "ab"),))
    assert one != two


def test_digest_empty_inputs_deterministic():
    assert digest("", ()) == digest("", ())
    assert len(digest("", ())) == 16


def test_digest_unfingerprintable_function_captures_closure():
    # When the strict walker aborts on an un-canonicalizable referenced value, the fallback must still capture
    # closure state: two functions with identical source but different closure values yield different digests. A
    # fallback that cloudpickled the bare __code__ would drop exactly that state.
    class Opaque:  # Opaque is not a canonicalizable type, so it forces the strict walker to abort.
        pass
    marker = Opaque()

    def make(k):
        def cb(x):
            return (marker, k, x)  # The tuple references the opaque global, and k is a closure variable.
        return cb

    assert digest("StateType", (make(1),)) != digest("StateType", (make(2),))


def test_digest_folds_dispatcher_jit_flags():
    # A Dispatcher's py_func identifies only part of the dispatcher, because numba compiles a different library for
    # each set of jit flags. Two dispatchers over one body must therefore produce distinct digests. The py_func
    # shortcut sees only the Python body, so it would give all of these variants a single digest.
    def cb(x):
        return x + 1

    variants = [
        njit(cb),
        njit(nogil=True)(cb),
        njit(fastmath=True)(cb),
        njit(error_model="numpy")(cb),
        njit(boundscheck=True)(cb),
    ]
    digests = [digest("StateType", (v,)) for v in variants]
    assert len(set(digests)) == len(digests)


def test_digest_of_dispatcher_is_deterministic_and_body_sensitive():
    def cb(x):
        return x + 1

    def other(x):
        return x + 2

    assert digest("StateType", (njit(cb),)) == digest("StateType", (njit(cb),))
    assert digest("StateType", (njit(cb),)) != digest("StateType", (njit(other),))


def test_digest_folds_dufunc_jit_flags():
    # @vectorize freezes its scalar body at compile time, so its targetoptions belong in the key too. This property
    # also holds on the cloudpickle path, so, unlike the Dispatcher case above, the test does not by itself verify
    # the _canon_value routing. For DUFuncs, the routing avoids cloudpickle's documented nondeterminism for __main__
    # objects.
    def cb(x):
        return x + 1

    plain = digest("StateType", (vectorize(cb),))
    checked = digest("StateType", (vectorize(boundscheck=True)(cb),))
    assert plain != checked
    assert plain == digest("StateType", (vectorize(cb),))


if __name__ == '__main__':
    collect_and_run_tests(__name__)
