numbox.core.configurations
==========================

Overview
++++++++

Every ``@njit``, ``@cres`` and ``@proxy`` in numbox is decorated under one set of numba options,
``jit_options``, read once from the ``NUMBOX_JIT_OPTIONS`` environment variable when this module is first
imported. The value is a JSON object passed to ``@njit`` as keyword arguments; unset means ``{"cache": true}``,
so numbox compiles into numba's on-disk cache by default, and ``export NUMBOX_JIT_OPTIONS='{"cache": false}'``
turns that off.

Where the cache lands
+++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory
(``~/.cache/numba`` on Linux), taking the first of those it can write. It sets the cache up when the function is
decorated, so with caching on the question of where it lands is settled at import.

Where no cache can be written, numba raises at decoration, and numbox's import used to die on its first module
with ``RuntimeError: cannot cache function ...: no locator available for file ...``. The two placements that do
that are a read-only install whose user cache directory cannot be written either, and an import from an
``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. Since every module here decorates
under the one ``jit_options``, numbox now puts the question once, when this module is imported, for a function
of its own, and answers it for the package: it runs the cache set-up numba runs at decoration and the
writability check numba runs at the first save, compiling nothing, and where either fails numbox compiles
without a cache and one ``RuntimeWarning`` names the remedy.

- For a source file on disk the remedy is ``NUMBA_CACHE_DIR`` pointed at a writable directory.
- For an archive it is an unpacked install or a ``.zip``: numba reads ``NUMBA_CACHE_DIR`` only for a source
  file on disk, so the variable changes nothing there, and a ``.zip`` is the one archive numba caches, from
  0.61 on, in the user's cache directory. numba takes that directory for a ``.zip`` without checking that it
  can be written, so a ``.zip`` with a read-only home falls back the same way.
- ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in either case.

A function numba cannot cache is compiled in every process that uses it, never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
