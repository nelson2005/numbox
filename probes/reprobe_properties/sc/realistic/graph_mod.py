"""A numbox Work graph with a mix of distinct and repeated derive expressions.

Repeating one expression across several nodes is ordinary graph authoring, and it
is the shape that mints one alias for several bodies.
"""
from numbox.core.work.builder import Derived, End, make_graph

n1 = End(name="n1", init_value=0.0)
n2 = End(name="n2", init_value=0.0)
f1 = Derived(name="f1", init_value=0.0, sources=(n1, n2), derive=lambda x, y: x + y)
f2 = Derived(name="f2", init_value=0.0, sources=(f1,), derive=lambda x: 2 * x)
f3 = Derived(name="f3", init_value=0.0, sources=(f2,), derive=lambda x: 2 * x)
f4 = Derived(name="f4", init_value=0.0, sources=(f3,), derive=lambda x: 2 * x)
f5 = Derived(name="f5", init_value=0.0, sources=(f4,), derive=lambda x: x - 1.0)
f6 = Derived(name="f6", init_value=0.0, sources=(f5,), derive=lambda x: x / 3.0)

GRAPH = make_graph(f6)
