numbox.core.vector
====================

Overview
++++++++

Generic growable numba vector backed by a numpy array.

Compared to ``numba.typed.List``:

- ``List`` supports arbitrary element types (including other structrefs)
  and exposes a richer API (``append``, ``pop``, ``insert``, ``remove``,
  slicing).
- ``Vector`` is restricted to scalar element types where ``str(elem_type)``
  matches a numpy dtype (``float64``, ``int64``, etc.).
  ``make_vector`` memoises instances by ``elem_type.key``, so cached code
  keeps the same type identity across processes. Storage is a single
  ``numpy.ndarray``, so per-element overhead is the scalar itself plus
  amortised geometric growth.
- ``AnyVector`` (prototype) lifts the scalar restriction by storing
  ``numbox.core.any.any_type.Any`` elements behind a contiguous buffer of
  NRT MemInfo pointers: one vector can hold values of mixed types, each
  recoverable via ``Any``'s type tag. Elements are reference-counted; a
  custom NRT destructor releases every remaining element when the vector
  itself dies, so dropping the vector cannot leak. See the module docstring
  for the full ownership contract.

Modules
++++++++

numbox.core.vector.vector
-----------------------------

.. automodule:: numbox.core.vector.vector
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.vector.any_vector
---------------------------------

.. automodule:: numbox.core.vector.any_vector
   :members:
   :show-inheritance:
   :undoc-members:
