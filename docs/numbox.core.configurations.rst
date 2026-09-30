numbox.core.configurations
==========================

Overview
++++++++

Every function numbox caches is decorated under one set of numba options, ``jit_options``, read once from
the ``NUMBOX_JIT_OPTIONS`` environment variable when this module is first imported; a bare ``@njit`` in
numbox, ``array_data_p`` or ``make_vector``'s ``create``, is not cached and takes none. The value is a JSON
object passed to ``@njit`` as keyword arguments, and any other shape is refused by name; unset means
``{"cache": true}``, so numbox compiles into numba's on-disk cache by default, and
``export NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns that off.

Where the cache lands
+++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory
(``~/.cache/numba`` on Linux), taking the first of those it can write. It sets the cache up when the function is
decorated, so with caching on the question of where it lands is settled at import.

Where no cache can be written, numba raises at decoration, and numbox's import used to die on its first module
with ``RuntimeError: cannot cache function ...: no locator available for file ...``. The two placements that do
that are a read-only install whose user cache directory cannot be written either, and an import from an
``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. Since every function numbox caches
decorates under the one ``jit_options``, numbox puts the question once, when this module is imported, and
answers it for the package: for a probe in each directory of the package that holds a module, since numba's
in-tree cache is a ``__pycache__`` beside each source, it runs the cache set-up numba runs at decoration and
the writability check numba runs at the first save, compiling nothing, and where either fails for any
directory numbox compiles without a cache and one ``RuntimeWarning`` names the remedy.

- For a source file on disk the remedy is ``NUMBA_CACHE_DIR`` pointed at a writable directory.
- For a ``.zip`` it is the user's cache directory made writable: a ``.zip`` is the one archive numba caches,
  from 0.61 on, and it caches it there, taking the directory without checking that it can be written. numba
  reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the variable changes nothing for any archive.
  A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, where numba
  alone would have loaded the entries: the writability check is the rule numba applies to every other
  placement.
- For an ``.egg``, ``.whl`` or ``.pyz`` it is an unpacked install or a ``.zip``.
- ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in every case.

A function numba cannot cache is compiled in every process that uses it, never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

The code numbox generates at run time, ``make_structref``'s, ``compile_kernel``'s, the work builder's derives
and the sqlite registrations', is anchored to a file under ``NUMBA_CACHE_DIR`` or the user's cache directory
so that numba can cache it. That directory can be unwritable where the package's own functions cache, since
those cache beside their sources, so each anchor puts the same question for its own file when it is written,
and the code it names compiles without a cache where the answer is no, after the same warning (the builder's
derive falls back without one, as it did). See the cache-anchor section of :doc:`numbox.utils`.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
