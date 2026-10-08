numbox.core.configurations
==========================

Overview
++++++++

Every function numbox caches is decorated under one set of numba options, ``jit_options``, read once from
the ``NUMBOX_JIT_OPTIONS`` environment variable when this module is first imported; a bare ``@njit`` in
numbox, as in ``lowlevel.py``, ``meminfo.py`` and ``make_vector``, is not cached and takes none. The value is a JSON
object passed to ``@njit`` as keyword arguments, and any other shape is refused by name, as is a ``cache``
that is not ``true`` or ``false`` (the string ``"false"`` is true to numba, which reads the option's truth);
unset means ``{"cache": true}``, so numbox compiles into numba's on-disk cache by default, and
``export NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns that off.

.. _where_the_cache_lands:

Where the cache lands
+++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory, taking the
first of those it can write. It sets the cache up when the function is decorated, so with caching on the question
of where it lands is settled at import. The user's cache directory is ``~/Library/Caches/numba`` on macOS,
``%LOCALAPPDATA%\numba\Cache`` on Windows, and ``$XDG_CACHE_HOME/numba`` elsewhere, ``~/.cache/numba`` when
``XDG_CACHE_HOME`` is unset.

Where no cache can be written, numba raises at decoration, and numbox's import used to die on its first module
with ``RuntimeError: cannot cache function ...: no locator available for file ...``. The two placements that do
that are a read-only install whose user cache directory cannot be written either, and an import from an
``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. Since every function numbox caches
decorates under the one ``jit_options``, numbox puts the question once, when this module is imported, and
answers it for the package: for a probe in each directory of the package that holds a module, since numba's
in-tree cache is a ``__pycache__`` beside each source, it runs the cache set-up numba runs at decoration and
the writability check numba runs at the first save, compiling nothing, and where either fails for any
directory numbox compiles without a cache and one ``RuntimeWarning`` names the remedy. Every directory
counts, whether or not its modules cache anything, so the answer errs toward uncached, which is never wrong.
A module that survives as ``.pyc`` alone is asked by the file it was compiled from, which its code keeps and
numba looks up: the ``.py`` that is gone where it was compiled in place, or a tree elsewhere, on disk or not.
For a ``.zip``, which numba caches per directory of the archive, each in a location of its own under the
user's cache directory, the archive's directories are listed and the question put for one module of each; a
``.pyc`` in the archive that zipimport would run, which it takes before the ``.py`` beside it unless it is
stale against it or of another interpreter, and whose code keeps the file it was compiled from, is asked by
that file, since that is what numba looks up for it, on disk or gone, and one zipimport would pass over is
passed over; any other archive has no location at all. A directory of the package reached through a symlink
is walked like the rest, wherever the link points. The check makes the cache directories it asks about, as
numba would at the first decoration in each; with caching beside the sources that is an empty ``__pycache__``
per directory of the package, a linked one included. numba's own writability check makes a temporary file,
one without a name on Linux, and the files it saves have names of a hundred bytes and more, so a location
within their length of the path limit, 4096 on Linux, passes numba's check and the first save overflows;
the check here makes a file named as long as the longest numba writes for the package's files (128 bytes,
which a test holds every function of the package under), so that location turns caching off with the
warning instead.

- For a source file on disk the remedy is ``NUMBA_CACHE_DIR`` pointed at a writable directory; where it is
  set and numba passed it over, the warning names it and asks for a writable directory at a short path, since
  numba passes over one too deep to make its directory in as it does an unwritable one; and where
  ``NUMBA_CACHE_LOCATOR_CLASSES``, from numba 0.62, leaves the user-provided locator out, it is the locations
  the list names made writable, numba never reading the variable. Where the
  location is too long for the file system, the remedy is for the location the error names: numba takes a
  directory under ``NUMBA_CACHE_DIR`` where that is set, else the ``__pycache__`` beside the source, else a
  directory under the user's cache directory, each of the two under a cache directory named for the source's
  directory, by its name and a hash of its path; the path the error names is matched whole against each location
  a locator numba reaches gives it, in the order numba tries them, since the three can nest and
  ``NUMBA_CACHE_DIR`` set to the user's cache directory makes one path of two; numba reaches no locator listed
  after its ``.zip`` one for a path with ``.zip`` in it, which that locator takes without trying its location, so
  a location one of those would give is none of numba's. So a shorter ``NUMBA_CACHE_DIR`` for the first; the
  package installed at a shorter path for the second; the user's cache directory at a shorter path, through
  ``XDG_CACHE_HOME`` or ``HOME``, ``HOME`` alone on macOS and nothing on Windows, where numba asks the system,
  for the third; and for either of the last two ``NUMBA_CACHE_DIR`` set to a short path, where numba tries it
  before the locator that took the location, or, where it is set and numba passed it over, unwritable or too
  deep, the warning names it and the remedy is a writable directory at a short path. A location that is none of
  numba's is named as the error names it, with the variable, where numba tries it before any other locator, named
  as set or asked for at a short path, and nothing said of numba passing it over, which that location cannot show.
  The name an error carries is read as a string, bytes or a path object alike. An error can name no file, or carry an
  empty string or empty bytes as its name. A caller can build such an error, and numba's first save raises one where a
  write fails, as on a full disk. So does inspect where, for numba's IPython locator, it finds no source in a cell file
  on disk. Every other OSError that ``check_cache_location`` raises names a file. An error with no name, or an empty
  one, is told the locations numba could have taken, in numba's order and each once. For a path too long, the remedy is
  to put whichever of them is too long at a shorter path. For a file refused for another reason, such as a full disk,
  the remedy gives the reason and asks for room or a writable directory there. A path object is never empty: pathlib
  reads an empty string as the working directory, which the remedy names as a location that is none of numba's.
  ``NUMBA_CACHE_LOCATOR_CLASSES`` decides that order, each entry a class of numba's caching module or, by its
  dotted path, a subclass of one, which takes what its parent takes, caches where it does and is told as it is; an
  entry numba cannot resolve, which it refuses before any location is tried, and one that names no class, which it
  fails on at a function it decorates where it reaches the entry, are no locators here, and numba's own error for
  either is raised as it was.
  numba's IPython locator caches a cell file it takes in ``numba_cache`` under IPython's cache directory, with no
  directory per file, and reads the function's source when it takes the file, which the check's probe is given
  under the file's name for a file not on disk that a locator of that family numba reaches takes, one check at a time,
  having no module to read a ``.zip`` member's through, where inspect reads a file on disk itself, and the
  linecache is left alone for every other file; that location too long is told as
  IPython's cache directory at a shorter path, for a cell file on disk and for a ``.zip`` member in an ipykernel
  directory alike, where that locator is listed before the ``.zip`` one, which takes a file without trying its
  location, so that none listed after it is reached, with no ``NUMBA_CACHE_DIR`` offered for a source not on
  disk; the remedy asks IPython for that directory where a locator of that family is listed
  and takes the file, and leaves IPython alone for every other file, a plain ``.zip`` member among them, since
  IPython warns when asked under a home it cannot write and leaves a temporary directory behind; numba
  imports IPython for the location when it makes it and catches only OSError there, so where IPython is not
  importable it raises ImportError at decoration for a file that locator takes, no cache error, and an error a
  caller passes on naming that location is told it as none of numba's, and where IPython raises OSError for its
  directory numba passes the locator over, as it does a location it cannot make, and the warning, asked the same,
  has no location of IPython's to match the error against. numba's IPython locator takes a file on disk
  only in an ipykernel directory and its ``.zip`` locator only a path with ``.zip`` in it, so either ahead of
  the rest changes nothing for any other file; the ``.zip`` locator ahead of them all takes such a path first and
  caches it under the user's cache directory where a part of the path ends in ``.zip``, an archive or a
  directory, or, where none does, finds no archive
  in it, and the warning then asks for a locator for a file on disk listed before it, the user-provided one with
  ``NUMBA_CACHE_DIR`` set, which alone it takes nothing without; after some of them, it finds no archive once
  those have passed the file over, numba trying none after it, and the warning asks for one of their locations
  made writable or such a locator listed before it. A list with no locator that takes the file is told so, for an
  error a caller passes on under it as for numba's no-locator one, and so is one whose only locator for the file is
  the user-provided one with ``NUMBA_CACHE_DIR`` unset, which takes nothing without it: the variable to set, as
  numba's no-locator error is told there. Where the location numba took refuses a file for another reason, a
  full disk or permissions changed since numba's own check, the warning names the location and the reason and
  asks for room or a writable directory there, with ``NUMBA_CACHE_DIR`` as the alternative where numba tries it
  before that location, or set to another directory where the location is the variable's own; numba passes over
  a location it cannot make or write in, so only its no-locator error means the variable was passed over. An
  error naming a location that is none of numba's is told that location as the error names it, and one naming
  no file is told the locations numba could have taken.
- For a ``.zip``, or a frozen application, it is the user's cache directory made writable, or room made there,
  with the reason the error gives, a full disk or permissions: a ``.zip`` is the one archive numba caches, from
  0.61 on, and it caches it there, taking the directory without checking that it can be written; the error names
  that location, or a file numba's first save writes in it, and the location is told with the reason; an error
  naming a location that is none of numba's, under the list and in its order, is told that location as the error
  names it, as for a source on disk, with nothing of what moves the user's cache directory and nothing of where
  numba caches the file, which that location is none of; a frozen
  application (``sys.frozen``) is cached there too, its sources not being on disk, and its error is numba's
  no-locator one, which gives no reason, numba having passed the location over on its error, unwritable or too deep
  alike, so that directory is named as one numba could not use, to be made writable or put at a shorter path.
  A frozen application's path with ``.zip`` in it and no part ending in it gets the ``.zip`` locator's error instead,
  where the list puts that locator before the user-wide one or once the user-wide one passed its location over, and
  the warning asks for the user-wide locator listed before it, or for that location made writable or put at a
  shorter path, as for a source on disk.
  numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the variable changes nothing for either.
  An error that names no file, a caller's, or numba's first save's on a full disk, is told the locations numba
  could have taken, as for a source on disk, in its order and each once, with the reason, or, for a path too long,
  to put whichever is too long at a shorter path; for a member in an ipykernel directory where a locator of
  IPython's family is reached, IPython's ``numba_cache`` is among them, and the warning then says nothing of the
  user's cache directory as where numba caches the file.
  A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, where numba
  alone would have loaded the entries: the writability check is the rule numba applies to every other
  placement. numba caches a ``.zip`` through its ``.zip`` locator alone, so a ``NUMBA_CACHE_LOCATOR_CLASSES``
  that leaves that locator out gives numba no locator for a source in a ``.zip``, and the warning asks for it to
  be listed; so for a frozen application and numba's user-wide locator, which alone takes one; either is told
  for an error a caller passes on under such a list as for numba's no-locator one, since numba took no location
  under it, and a source neither in a ``.zip`` nor a frozen application's, which no locator takes, is told the
  source files on disk or a ``.zip`` holding them for any error.
- For an ``.egg``, ``.whl`` or ``.pyz``, a ``.pyc``-only install or a ``.pyc`` in a ``.zip``, it is the source
  files on disk or a ``.zip`` holding them, each with the locator it needs listed where
  ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it out: one of the three locators for a file on disk for the first, the
  ``.zip`` locator for the second, and ``NUMBA_CACHE_DIR`` set where the user-provided locator is the one listed for
  a file on disk, which takes nothing without it.
- ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in every case, the
  package's options being what it sets; the anchors' warning under a caller's own options, below, is
  silenced by those.

A function numba cannot cache is compiled in every process that uses it, never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

A package built on numbox can put the same question for its own files: ``check_cache_location`` asks it for
one file, ``is_a_cache_error`` tells numba's cache errors from the rest, and ``cache_remedy`` words the
remedy, given the package's name for the two remedies that tell the reader to install it again, at a shorter
path or with its source files on disk. The bound on file names is the package's own:
``LONGEST_CACHE_FILE_NAME`` holds numbox's functions only, so a package passes ``check_cache_location`` the
longest name numba writes for its own, as ``longest_file_name``, or a location that fits numbox's names and not
the package's passes the check and the first save overflows it.

The code numbox generates at run time, ``make_structref``'s, ``compile_kernel``'s, the work builder's derives
and the sqlite registrations', is anchored to a file under ``NUMBA_CACHE_DIR`` or the user's cache directory
so that numba can cache it. That directory can be unwritable where the package's own functions cache, since
those cache beside their sources, so each anchor puts the same question for its own file when it is written,
and the code it names compiles without a cache where the answer is no, after a warning of the same shape (the
builder's derive falls back without one, as it did). ``make_structref``, ``compile_kernel`` and the builder
take jit options of the caller's, which the variable does not reach, and ``compile_kernel``'s ``cache``
argument overrides those too, so that warning's silence is ``cache`` off in the options the code was given,
the argument where it takes one, or the variable where the options are the package's. ``make_graph``'s
kernel is anchored to the builder's own file and cached beside it, so it puts the question for that file
under the options it was given, and falls back the same way with the package's remedy for the placement.
See the cache-anchor section of :doc:`numbox.utils`.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
