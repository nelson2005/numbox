numbox.core.any
===============

Overview
++++++++

Implementation of `Any` type counterpart of C++
`std::any <https://en.cppreference.com/w/cpp/utility/any.html>`_
leveraging the type-erasure technique.

`FrozenAnyTuple` is an immutable, build-once / read-many heterogeneous snapshot: a
structref-wrapped ``UniTuple`` of `Any` carrying per-slot type codes, supplying
container-level type discipline now that `Any` itself performs no check on reads.
Prefer it over ``numba.typed.List`` for frozen heterogeneous data: its reads are
faster at every measured arity, and unlike ``List``'s uncacheable per-process
container compilation tax, its compile cost is one-time under ``cache=True``.
Prefer `Vector` when the data is growable and of a single numpy scalar type;
`FrozenAnyTuple` is heterogeneous and frozen.

Modules
++++++++

numbox.core.any.any\_type
-------------------------

.. automodule:: numbox.core.any.any_type
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.any.frozen\_any\_tuple
----------------------------------

.. automodule:: numbox.core.any.frozen_any_tuple
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.any.content\_wrap
-----------------------------

.. automodule:: numbox.core.any.content_wrap
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.any.erased\_type
----------------------------

.. automodule:: numbox.core.any.erased_type
   :members:
   :show-inheritance:
   :undoc-members:
