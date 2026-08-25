"""Does the content fingerprint separate bodies that differ only in a captured value?

Both alias minters fold ``_body_fingerprint``. If two materially different bodies collapse onto one
digest, a warm caller compiled against the first silently runs the second, and the guard sees a
resolving alias and passes the object.
"""
from numba import njit, types
from numba.core.types import ExternalFunction

from numbox.utils.fingerprint import _body_fingerprint
from numbox.utils.derive_wap import _stable_jit_alias

SIG = types.float64(types.float64)


def make_extern(sym):
    ext = ExternalFunction(sym, SIG)

    def body(x):
        return ext(x)
    return body


def make_literal(k):
    def body(x):
        return x * k
    return body


def make_global_ref(fn):
    def body(x):
        return fn(x)
    return body


@njit(SIG)
def helper_a(x):
    return x * 2.0


@njit(SIG)
def helper_b(x):
    return x * 3.0


cases = {
    "extern cov_scale vs cov_other": (make_extern("cov_scale"), make_extern("cov_other")),
    "closure literal 2.0 vs 3.0": (make_literal(2.0), make_literal(3.0)),
    "closed-over dispatcher a vs b": (make_global_ref(helper_a), make_global_ref(helper_b)),
    "identical extern (control)": (make_extern("cov_scale"), make_extern("cov_scale")),
}

for label, (f1, f2) in cases.items():
    fp1, fp2 = _body_fingerprint(f1), _body_fingerprint(f2)
    a1 = _stable_jit_alias(f1, SIG, {"cache": True})
    a2 = _stable_jit_alias(f2, SIG, {"cache": True})
    verdict = "SAME (collision)" if a1 == a2 else "distinct"
    if label.endswith("(control)"):
        verdict = "SAME (expected)" if a1 == a2 else "DIFFERENT (unexpected)"
    print(f"{label:34s} fp_equal={fp1 == fp2!s:6s} alias_equal={a1 == a2!s:6s} -> {verdict}")
    print(f"    a1={a1}\n    a2={a2}")
