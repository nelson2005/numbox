numbox.core.configurations
==========================

Overview
++++++++

Every function numbox caches is decorated under one set of numba options, ``jit_options``. numbox reads them once
from the ``NUMBOX_JIT_OPTIONS`` environment variable when this module is first imported. A bare ``@njit`` in numbox,
as in ``lowlevel.py``, ``meminfo.py`` and ``make_vector``, is not cached and takes none of those options. The value
is a JSON object, passed to ``@njit`` as keyword arguments. numbox refuses any other shape by name, and refuses a
``cache`` that is not ``true`` or ``false``: the string ``"false"`` is true to numba, which reads the option's truth.
An unset variable means ``{"cache": true}``, so numbox compiles into numba's on-disk cache by default, and
``export NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns that off.

.. _where_the_cache_lands:

Where numba writes the cache
++++++++++++++++++++++++++++

numba writes a cached function's entries under ``NUMBA_CACHE_DIR`` when that variable is set, else into the
``__pycache__`` directory beside the function's source file, else into the user's cache directory. It takes the
first of those it can write.
numba sets the cache up when the function is decorated, so with caching on the location is fixed at import.
The user's cache directory is ``~/Library/Caches/numba`` on macOS,
``%LOCALAPPDATA%\numba\Cache`` on Windows, and ``$XDG_CACHE_HOME/numba`` elsewhere, or ``~/.cache/numba`` when
``XDG_CACHE_HOME`` is unset.

The check at import
-------------------

Where no cache can be written, numba raises at decoration. Before this check, numbox's import failed on its first
module with ``RuntimeError: cannot cache function ...: no locator available for file ...``. Two placements cause
that: a read-only install whose user cache directory cannot be written either, and an import from an ``.egg``,
``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships.

Every function numbox caches is decorated under the one ``jit_options``, so numbox checks once, when this module is
imported, and the answer holds for the package.
numba's in-tree cache is a ``__pycache__`` beside each source, so the check makes a probe for each directory of the
package that holds a module: a function, never compiled, whose code names the file being checked.
The check runs, for each probe, the cache set-up numba runs at decoration and the writability check numba runs at the
first save, and compiles nothing.
Where
either fails for any directory, numbox compiles without a cache, and one ``RuntimeWarning`` names the remedy.
Every directory counts, whether or not its modules cache anything, so the answer leans toward uncached, which is
never wrong.

A module that exists as a ``.pyc`` alone is checked by the file it was compiled from.
Its code keeps that file's name, and numba looks the file up: the ``.py`` that no longer exists where the module was
compiled in place, or a tree elsewhere, on disk or not.

numba caches a ``.zip`` per directory of the archive, each in a location of its own under the user's cache
directory, so the check lists the archive's directories and checks one module of each.
A ``.pyc`` in the archive that zipimport would run is checked by the file it was compiled from, which its code keeps
and numba looks up, on disk or absent.
zipimport runs the ``.pyc`` before the ``.py`` beside it, unless the ``.pyc`` is stale against the ``.py`` or belongs
to another interpreter; the check also passes over a ``.pyc`` that zipimport would pass over.
A
``.pyc`` whose code names a member the archive does not hold is checked for itself, whichever member stands for its
directory.
Such a ``.pyc`` is one with no ``.py`` beside it, or one compiled with ``compileall -d`` to name another path in the
archive, with or without a ``.py`` beside it.
numba 0.68 and later read the member's source from the archive by the name its code gives, and find no source for it.
Any other archive has no location at all.
A directory of the package reached through a symlink is walked like the other directories, wherever the link points.

The check makes the cache directories it asks about, as numba would at the first decoration in each.
With caching beside the sources, that is an empty ``__pycache__`` per directory of the package, including a linked
one.

numba's own writability check makes a temporary file, one without a name on Linux, while the files numba saves have
names of a hundred bytes and more. A location within that length of the path limit, 4096 on Linux, passes numba's
check, and the first save then overflows the limit.
The check here makes a file named as long as the longest name numba writes for the package's files, 128 bytes; a test
keeps every function of the package under that length.
Such
a location therefore turns caching off with the warning instead.

A function numba cannot cache is compiled in every process that uses it, and is never wrong; that is the cost the
warning reports. An error at decoration that is not the cache's is raised as it was.

The remedies
------------

The warning names a remedy, chosen by the placement of the source and by the error; some cases offer alternatives.

A source file on disk
~~~~~~~~~~~~~~~~~~~~~

**A file numba cannot read.** numba's IPython locator reads the source file for the function's source, and from
numba 0.68 every locator reads it for the stamp. The remedy is to make the file readable.

**A file numba can read.** The remedy is ``NUMBA_CACHE_DIR`` pointed at a writable directory.
Where ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves the user-provided locator out, it is the locations the list names made
writable, numba never reading the variable.
The warning names the variable where it is set and numba passed it over, and asks for a writable directory at a short
path. numba passes over a directory too deep to make its cache directory in, as it passes over an unwritable one.

**A location too long for the file system.** The remedy is for the location the error names. numba takes a
directory under ``NUMBA_CACHE_DIR`` where that is set, else the ``__pycache__`` beside the source, else a directory
under the user's cache directory.
The directory numba takes under ``NUMBA_CACHE_DIR`` or under the user's cache directory is a cache directory named
for the source's directory, by its name and a hash of its path.
The warning matches the path the error names, as a whole, against each location that a locator numba reaches gives
the file, in the order numba tries the locators.
It matches the whole path because the three locations can nest. It takes the locators in numba's order because
``NUMBA_CACHE_DIR`` set to the user's cache directory makes one path of two.
The warning attributes the error to the first locator whose location matches, taking the locators numba tries before
it to have passed the file over.
numba's own errors carry that meaning; an error a caller passes on, naming a location numba did not take, need not.
For a path with ``.zip`` in it, numba reaches no locator listed after its ``.zip``
locator, which takes the path without trying its location, so a location that one of those would give is none of
numba's.

The remedy for a location under ``NUMBA_CACHE_DIR`` is a shorter ``NUMBA_CACHE_DIR``. The remedy for the
``__pycache__`` beside the source is the package installed at a shorter path.
The remedy for the user's cache directory is that directory at a shorter path, through ``XDG_CACHE_HOME`` or
``HOME``; on macOS ``HOME`` alone moves it, and on Windows no variable moves it, because numba asks the system for
it.
For either of the last two, the remedy
is also ``NUMBA_CACHE_DIR`` set to a short path, where numba tries the variable before the locator that took the
location.
The warning names the variable where it is set and numba passed it over, unwritable or too deep, and asks for a
writable directory at a short path.

**A directory name too long for the file system.** The directory numba names after the source's directory, under
``NUMBA_CACHE_DIR`` or the user's cache directory, is that directory's name and forty-one bytes.
The file system holding the location bounds a name's length; the warning asks the system for the bound, and takes 255
if the system cannot report it.
The bound is in bytes, or in the UTF-16 units that Windows and macOS count a name in.
A name longer than the bound can be made at no path, and the system refuses it regardless of the errno, EINVAL on
Windows.
The warning gives the name and its length, and asks for the file in a directory of a shorter name, for three errors.
The first is an error naming such a location.
The second is numba's no-locator error for a file on disk, where every locator numba tried that names its directory
in that way found the name too long for its file system.
For the variable's locator, the warning measures by the limit of the directory the variable is set to, or by 255 if
it is unset.
The third is a
frozen application's error.
The warning also asks, for a file on disk, for the ``__pycache__`` beside the source made writable where the in-tree
locator is listed, or for that locator listed where it is not. The in-tree locator caches under no such name.

**A cache file name too long for the file system.** numba names a cache file after the source's name and the
function's qualified name.
numba can fail to make or open such a file in the location it took because the name is longer than the file system
allows.
The warning then gives the name and its length, which a shorter path does not remove, and asks for a shorter name for
the function or its file; that name is the package's to give.
The warning for the check's own error, for a name as long as the longest the package's functions get, on a file
system that takes no name that long, states the length.
The warning offers ``NUMBA_CACHE_DIR`` pointed at a directory on a file system that takes the name, where such a file
system is in use and numba tries the variable before the locator that took the file.
It offers the variable on the same terms as the short-path offer for a path too long, when numba tries the variable
before the locator that took the file.

A name fits such a file system when its base and the 29 bytes of the longest name numba writes for the function are
within the 255 those file systems allow. That name belongs to a data file, with three digits for its number; numba
counts the data files from one without bound, and writes each file through a temporary name.
The warning offers no ``NUMBA_CACHE_DIR`` for a ``.zip`` member, for which the variable has no effect.

**A directory path that is not UTF-8.**
numba hashes the source's directory path as UTF-8 for that cache directory name, and raises at decoration for bytes
that are not UTF-8, where a locator that names its directory in that way reaches the file.
For such a path, the remedy names no location under ``NUMBA_CACHE_DIR`` or the user's cache directory.
numba's ``.zip`` locator hashes the archive's path the same way for its directory under the user's cache directory,
so the warning for a ``.zip`` under such a path asks for the archive under a UTF-8 path.
numba's user-wide locator hashes a frozen application's path the same way, so the warning for a frozen application
under such a path asks for the application under a UTF-8 path.
The warning for a file on disk under such a path also names the ``__pycache__`` beside it, since numba's in-tree
locator hashes no path.
For a file on disk under such a path, the remedy is one of three.
List that locator before the first that hashes, if the list leaves it out or puts it after such a locator.
Unset ``NUMBA_CACHE_DIR`` if its locator alone hashes before the in-tree one.
Or make the ``__pycache__`` beside the source writable if the in-tree locator came first and passed it over. With any
of these, that ``__pycache__`` is to be made writable where it cannot be written.

**A location that is none of numba's.** The warning names such a location as the error names it.
The warning names ``NUMBA_CACHE_DIR`` as set, or asks for it at a short path, where numba tries the variable before
any other locator; it does not say that numba passed the variable over, which such a location cannot show.

**What the error names.**
The warning reads the name an error carries as a string, whether the error carries a string, bytes or a path object,
and reads a name of another kind, an integer file descriptor's name, as none.
An error can name no file, or carry
an empty string or empty bytes as its name. A caller can build such an error, and numba's first save raises one
where a write fails, as on a full disk.
inspect raises one as well where, for numba's IPython locator, it reads no source in a cell file on disk, because the
file is unreadable or, before Python 3.13, empty. That error carries no errno, since no call of the file system
raised it.

The warning reads that error as inspect's error for a file on disk that a locator of IPython's family numba reaches
takes, since that locator tried no location. It asks that the file be made readable, with the source in it.
For a cell, numba reads the source from linecache, where IPython holds it while it runs the cell, and caches the cell
only while IPython holds its source; the warning states this.
The warning reads the error as the archive's for a ``.zip`` member whose archive holds no source, a ``.pyc`` alone,
and asks for its source files to be shipped.
Every other OSError that ``check_cache_location`` raises names a file.
The warning for an error with no name, or an empty one, gives the locations numba could have taken, in numba's order
and each once.
For a path too
long, the remedy is to put whichever of them is too long at a shorter path. For a file refused for another reason,
such as a full disk, the remedy gives the reason and asks for room or a writable directory there. A path object is
never empty: pathlib reads an empty string as the working directory, which the remedy names as a location that is
none of numba's.

**The locator list.** ``NUMBA_CACHE_LOCATOR_CLASSES`` decides the order in which numba tries locators. Each entry
is a class of numba's caching module, or, by its dotted path, a subclass of one, or a class of none of numba's
families.
A subclass takes the files its parent takes, caches where its parent caches, and the warning treats it as its parent.
A
class of none of numba's families has a ``from_function`` and a place of its own.
It takes the files its ``from_function`` takes when asked with a probe, as numba asks it, and only the class knows
where it caches.
For such a class, the warning for an error naming a location that no locator of numba's gives the file names that
location as the error names it, and treats the location as that class's.
The warning for an error naming no file gives the location numba took as the class's own, with no path to name, among
the places of numba's locators where those are reached.
Three kinds of entry are not locators for the warning, and numba's own error for them is raised unchanged.
The first is an entry
numba cannot resolve: numba refuses it before any location is tried, or raises importlib's error for it, a relative
module path's TypeError.
The second is an entry that names no class, and the third a class with no ``from_function``; numba fails on either at
a function it decorates where it reaches the entry.

**The IPython locator.** numba's IPython locator caches a cell file it takes in ``numba_cache`` under IPython's
cache directory, with no directory per file. It reads the function's source when it takes the file, as inspect
reads it: a file on disk itself, a cell from linecache, and a ``.zip`` member through the module's loader.
The check's probe has such a loader. The probe is a function of a module of its own, whose loader reads the member
from the archive as zipimport's does, and the member's lines are then in linecache as any reader leaves them.
Where that location is
too long, the warning asks for IPython's cache directory at a shorter path, for a cell file on disk and for a
``.zip`` member in an ipykernel directory alike. It does so where that locator is listed before the ``.zip`` one,
which takes a file without trying its location, so that no locator listed after it is reached; it offers no
``NUMBA_CACHE_DIR`` for a source not on disk.
A cell, which no other locator of numba's takes, receives numba's no-locator error where that locator passes its
location over.
The warning then names that location, to make writable or put at a shorter path, or names IPython's cache directory,
to be given a home, where IPython raised for it.
A class of none of numba's families listed with that locator is
named as having passed the cell over too.
The remedy asks IPython for that directory only where a locator of that family is listed and takes the file, and does
not ask IPython for any other file, a plain ``.zip`` member among them. The reason is that IPython warns when asked
under a home it cannot write, and leaves a temporary directory behind, a new one on every call where it can write
neither its cache directory nor its own.
numba then makes ``numba_cache`` in one temporary directory and saves in another, and the check raises the error
numba's first save would raise, a ``FileNotFoundError`` naming the directory.
For that error, the warning asks for ``IPYTHONDIR``, or ``XDG_CACHE_HOME`` on a POSIX system, where IPython 8 and
later read it, at a writable path.
An error of another
errno naming a ``numba_cache`` comes from a caller, and the warning names the location it names.
numba imports IPython for the location when it makes the directory, and catches only OSError there.
If IPython is not importable, numba raises ImportError at decoration for a file that locator takes. That error is not
a cache error, and the check raises it as it was.
With IPython not importable, the warning for an error a caller passes on, naming that location, names the location as
none of numba's where another locator numba reaches has a place for the file. Where none has, it names the list, as
it does for numba's no-locator error.
Where IPython raises OSError for its directory, numba passes the locator over, as it does a location it cannot make,
and the warning, which asks IPython the same question, has no location of IPython's to match the error against.

**The order of the locators.**
numba's IPython locator takes a file on disk only in an ipykernel directory, and its ``.zip`` locator only a path
with ``.zip`` in it, so placing either before the others changes nothing for any other file.
The ``.zip`` locator placed before all the others takes such a path first.
It caches the file under the user's cache directory where a part of the path ends in ``.zip``, an archive or a
directory.
Where no part does, it finds no archive in the path, and the warning asks for a locator for a file on disk listed
before it; the user-provided one takes nothing without ``NUMBA_CACHE_DIR`` set.

For a path with no part ending in ``.zip``, the ``.zip`` locator listed after some of the others finds no archive
once those have passed the file over, and numba tries none after it. The warning then asks for one of their locations
made writable, or for a locator for a file on disk listed before it.
The warning for a list with no locator that takes the file says so, for an error a caller passes on under it as for
numba's no-locator one.
It says the same for a list whose only locator for the file is the user-provided one with ``NUMBA_CACHE_DIR`` unset,
which takes nothing without it. The remedy is the variable set, as for numba's no-locator error there.

**Another reason.**
The location numba took can refuse a file for another reason, a full disk or permissions changed since numba's own
check. The warning then names the location and the reason, and asks for room or a writable directory there.
It offers ``NUMBA_CACHE_DIR`` as the alternative where numba tries the variable before that location, or offers the
variable set to another directory where the location is the variable's own.
numba passes over a
location it cannot make or write in, so only its no-locator error means the variable was passed over.
The warning for an error naming a location that is none of numba's names that location as the error names it; the
warning for one naming no file gives the locations numba could have taken.

A ``.zip`` or a frozen application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The remedy is to make the user's cache directory writable, or to make room there, with the reason the error gives, a
full disk or permissions. A ``.zip`` is the one archive numba caches and it caches it there, taking the directory
without checking that it can be written. The error names that location, or a file numba's first save writes in it,
and the warning gives the location with the reason.
The warning for an error naming a location that is none of numba's, under the list and in its order, names that
location as the error names it, as for a source on disk.
The warning then says nothing of what moves the user's cache directory, which moves nothing of that location, and
nothing of where numba caches the file, since that location is none of numba's.

A frozen application (``sys.frozen``) is cached there too, since its sources are not on disk.
Its error is numba's no-locator one, which gives no reason, because numba passed the location over on its error,
whether the location was unwritable or too deep.
The warning therefore names that directory as one numba could not use, to be made writable or put at a shorter path.
The warning instead says that the name is too long if the name numba gives the directory after the source's is too
long for the file system.
It says the same of a cache file's name too long for the file system, in that location or in IPython's
``numba_cache`` for a member in an ipykernel directory, which a shorter path does not remove.

A frozen application's path with ``.zip`` in it and no part ending in it gets the ``.zip`` locator's error instead,
where the list puts that locator before the user-wide one, or once the user-wide one passed its location over. The
warning then asks for the user-wide locator listed before it, or for that location made writable or put at a shorter
path.
numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk, so the variable changes nothing for a ``.zip``
or a frozen application.

The warning for an error that names no file, a caller's or numba's first save's on a full disk, gives the locations
numba could have taken, as for a source on disk. It gives them in numba's order and each once, with the reason.
For a path too long, the remedy is to put whichever is too long at a shorter path.
The warning for such an error on a member in an ipykernel directory, where a locator of IPython's family is reached,
says nothing of the user's cache directory as the place numba caches the file. IPython's ``numba_cache`` is among the
locations it gives where IPython gives its directory.

A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, where numba alone
would have loaded the entries: the writability check is the rule numba applies to every other placement. numba
caches a ``.zip`` through its ``.zip`` locator alone, so a ``NUMBA_CACHE_LOCATOR_CLASSES`` that leaves that locator
out gives numba no locator for a source in a ``.zip``, and the warning asks for it to be listed. The same holds for
a notebook cell and numba's IPython locator, and for a frozen application and numba's user-wide locator, which
alone takes one.
The warning says so for any of them, for an error a caller passes on under such a list, as for numba's no-locator
one, since numba took no location under the list.
The warning for a source neither in a ``.zip`` nor a frozen application's, which no locator takes, asks for the
source files on disk, or a ``.zip`` holding them, for any error.

An archive or a module without its source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an ``.egg``, ``.whl`` or ``.pyz``, a ``.pyc``-only install, or a ``.pyc`` in a ``.zip``, the remedy is the source
files on disk, or a ``.zip`` holding them. Each needs its locator listed where ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves
it out: one of the three locators for a file on disk for the first, and the ``.zip`` locator for the second.
``NUMBA_CACHE_DIR`` must be set where the user-provided locator is the one listed for a file on disk, since that
locator takes nothing without the variable.

Archive stamps from numba 0.68
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From numba 0.68, the archive's stamp hashes each member's source, read from the archive by the name the module's
code gives under the archive's path.
The warning for a member shipped as ``.pyc`` alone asks for the source files, on disk or in a ``.zip``, and names the
member.
numba 0.68.0 on Windows refuses every member, because it builds the member's name with the platform's separator
(`numba issue 10889 <https://github.com/numba/numba/issues/10889>`_, fixed for 0.68.1); the warning asks for the
install on disk, or a release that contains the fix.
numba treats a directory whose name ends in ``.zip`` above the archive as the archive, by the first such part of the
path, and from 0.68 opens it for the source.
The warning says that such a directory is a directory, not an archive, and asks for the archive under a path with no
such directory above it.
The warning for a source file on disk under such a directory, which that locator takes once the locators before it
passed the file over, asks to make one of their locations writable, or to put the file elsewhere.

Turning caching off
~~~~~~~~~~~~~~~~~~~

``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning in each of the cases above,
because it sets the package's options. The anchors' warning under a caller's own options, below, is silenced by those
options.

Packages built on numbox
------------------------

A package built on numbox can run the same check for its own files: ``check_cache_location`` checks one file,
``is_a_cache_error`` distinguishes numba's cache errors from the rest, and ``cache_remedy`` phrases the remedy.
``cache_remedy``
takes the package's name, for the two remedies that tell the reader to install the package again, at a shorter path
or with its source files on disk. The bound on file names is the package's own. ``LONGEST_CACHE_FILE_NAME`` holds
numbox's functions only, so a package passes ``check_cache_location`` the longest name numba writes for its own
functions, as ``longest_file_name``. Otherwise a location that fits numbox's names and not the package's passes the
check, and the first save overflows it.

Generated code and its anchors
------------------------------

The code numbox generates at run time, by ``make_structref``, by ``compile_kernel``, for the work builder's derives
and for the sqlite registrations, is anchored to a file under ``NUMBA_CACHE_DIR`` or the user's cache directory, so
that numba can cache it. That directory can be unwritable where the package's own functions cache, since those
cache beside their sources.
Each anchor therefore runs the same check for its own file when the file is written, and the code it names compiles
without a cache where the check fails, after a warning of the same form.
The builder's derive falls back without a
warning, as it did.
``make_structref``, ``compile_kernel`` and the builder take the caller's jit options, which the variable does not
affect, and ``compile_kernel``'s ``cache`` argument overrides those too.
That warning is silent when ``cache`` is off in the options the code was given, in the
argument where the code takes one, or in the variable where the options are the package's. ``make_graph``'s kernel
is anchored to the builder's own file and cached beside it, so it runs the check for that file under the options it
was given, and falls back the same way, with the package's remedy for the placement. See the cache-anchor section of
:doc:`numbox.utils`.

Modules
++++++++

numbox.core.configurations
--------------------------

.. automodule:: numbox.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
