"""A Work built inside jitted scope from a @proxy binding's .as_func reached as
a compile-time constant, plus the same from a cres derive."""
from numba import njit, float64
from numbox.core.proxy.proxy import proxy
from numbox.core.work.work import make_work
from numbox.utils.highlevel import cres


@proxy(float64(float64))
def wp_body(x):
    if x > 0:
        raise ValueError("work proxy boom")
    return x * 2.0


@cres(float64(float64))
def wc_body(x):
    if x > 0:
        raise ValueError("work cres boom")
    return x * 2.0


WP = wp_body.as_func
WC = wc_body.as_func if hasattr(wp_body, "nope") else wc_body


@njit(cache=False)
def build_and_calc_proxy(v):
    src = make_work("src", v, (), None)
    node = make_work("node", 0.0, (src,), WP)
    node.calculate()
    return node.data, node.derived


@njit(cache=False)
def build_and_calc_cres(v):
    src = make_work("src", v, (), None)
    node = make_work("node", 0.0, (src,), WC)
    node.calculate()
    return node.data, node.derived


for label, fn, v in (("WORKCONST proxy raise", build_and_calc_proxy, 4.0),
                     ("WORKCONST proxy ok", build_and_calc_proxy, -4.0),
                     ("WORKCONST cres raise", build_and_calc_cres, 4.0),
                     ("WORKCONST cres ok", build_and_calc_cres, -4.0)):
    try:
        print("%-24s -> %r" % (label, fn(v)))
    except Exception as e:
        print("%-24s RAISED %s: %s" % (label, type(e).__name__, str(e)[:50]))
