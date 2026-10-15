numbox.core.cache_key
=====================

Overview
++++++++

numba keys a cached function on its signature, the target machine, and hashes of its bytecode and closure. Four
things that change the binary numba builds are not in that key: the jit flags (``fastmath``, ``error_model``,
``boundscheck`` and the rest), the ``NUMBA_*`` settings that change the code numba emits, the values of the globals
the function reads, and the code of the functions it calls. Change one of them between two runs and the second run
is served the first run's binary, with nothing said.

numbox adds all four to the key of every function it compiles, its bindings, structrefs, kernels, derives and the
functions passed to ``@proxy`` and ``@cres`` among them, and of any function decorated with
:func:`~numbox.core.cache_key.keyed_njit` or :func:`~numbox.core.cache_key.keyed_cfunc`. Everything else keeps
numba's key.

.. code-block:: python

    import settings
    from numbox.core.cache_key import keyed_njit


    @keyed_njit(cache=True)
    def scaled(x):
        return x * settings.SCALE

Run that with ``settings.py`` holding ``SCALE = 2.0``, change it to ``3.0`` and run it again. numba stamps a
function's cache with the function's own file, which has not changed, so under ``numba.njit`` the second run loads
the first run's binary and still multiplies by two; under ``keyed_njit`` the key holds ``settings.SCALE``, so the
second run misses and compiles anew. Both entries stay in the cache, so a third run with ``SCALE = 2.0`` loads the
first again.

``keyed_njit`` and ``keyed_cfunc`` take ``numba.njit``'s and ``numba.cfunc``'s arguments unchanged, signatures and
options included, and without ``cache=True`` they are those decorators. The key is set up when numba sets the cache
up, at decoration, so it applies to an eager compile, a signature given, as to a lazy one. It belongs to the cache
numba sets up for that decoration, not to the function: a plain ``numba.njit`` of the same function elsewhere keeps
numba's key. ``@vectorize``, ``@guvectorize`` and CUDA kernels keep numba's key.

What the key holds
++++++++++++++++++

Past numba's own key, the key of a function numbox keys holds these parts, each with a digest of its own:

- **numbox's version.** numbox's ``@overload`` and ``@intrinsic`` code generators shape the code of whatever calls
  them, and nothing else in the key can see into them.
- **The flags.** The decorator, ``njit`` or ``cfunc``, which numba keys alike; its options; its ``locals``; and its
  pipeline class.
- **The settings.** Each keyed ``NUMBA_*`` setting (see `The settings`_), twice: its value when numba built its code
  generator, and its value now.
- **The code, all the way down.** The function's bytecode, constants, default arguments and closure cells, and the
  same for every function it calls, followed through its globals, its closure cells, its default arguments and the
  module attributes it reads.
- **The values it reads.** Every global and module attribute each of those functions reads. Arrays are hashed in
  full. A namedtuple goes in with its class and field names, since numba compiles a read of a field to its position.
  A ``@proxy`` binding goes in by its alias, which names its body, signature and options.

Some values have no content the key can read, and go in by name: an ``@intrinsic``, with its definition's own code
and closure; a numba type; a class; a ufunc; a builtin; a callable numba types through a registered ``@overload``,
whose body numba never compiles, numpy's array functions such as ``np.sum`` among them; and a context numba compiles
a ``with`` block over, such as ``objmode``, whose block is the function's own code. Each goes in with the version of
the installed package that defines it. A ``register_jitable`` function is the exception among the overloaded: numba
compiles its body, so it is followed like any other function. A class is the exception where numba compiles what it
holds into the code that uses it: a namedtuple class goes in with its field names, an ``Enum`` class with its
members' values, and a jitclass with its field types and the code of its methods. An ``@intrinsic`` or ``@overload``
of your own code, from a module no installed package provides, has no version to stand in for its code generator:
it is followed like any function, so a change to it, or to a value it reads, misses.

The values are read when numba looks the function up, which is when it is first called with a signature, or when
it is decorated with one. A global or a setting changed in code after import counts for anything compiled
afterwards, and so does a ``NUMBA_*`` variable set in ``os.environ``: numba reads those again as a compile starts,
and the key reads them as that compile will. A function numba has already compiled keeps its binary for the life of the process, whatever changes
afterwards: that is numba's behaviour, and the key on disk names what the binary was built under.

The settings
++++++++++++

A setting is keyed if flipping it changes the LLVM IR or assembly numba emits: a survey of numba 0.60 to 0.68, each
minor at its latest patch, compiled a fixed set of functions under each setting flipped and compared the code.
Settings that change only speed or debugging information count, as do ``NUMBA_DUMP_ASSEMBLY``, ``NUMBA_DEBUG`` and
``NUMBA_DIFF_IR``, which change the code as a side effect of the dumps they ask for. Three change it only beside
another setting, which a flip of one setting at a time cannot show, and were found in numba's source:
``NUMBA_DEBUG_NRT_STACK_LIMIT`` under ``NUMBA_DEBUG_NRT``, and ``NUMBA_FULL_TRACEBACKS`` and ``NUMBA_DEVELOPER_MODE``
in a bounds check. The keyed settings are :data:`~numbox.core.cache_key.KEYED_SETTINGS`:

=================================== ===================================================================
Setting                             What flipping it changes
=================================== ===================================================================
``NUMBA_BOUNDSCHECK``               bounds checks on indexing
``NUMBA_OPT``                       every function, ``max`` included
``NUMBA_LOOP_VECTORIZE``            vectorised loops
``NUMBA_SLP_VECTORIZE``             vectorised straight-line code
``NUMBA_ENABLE_AVX``                instruction selection
``NUMBA_DEBUGINFO``                 debug information in every function
``NUMBA_ENABLE_PROFILING``          the same, as ``NUMBA_DEBUGINFO``'s default
``NUMBA_EXTEND_VARIABLE_LIFETIMES`` every function
``NUMBA_LLVM_REFPRUNE_PASS``        reference count pruning
``NUMBA_LLVM_REFPRUNE_FLAGS``       which pruning runs
``NUMBA_DEBUG_NRT``                 runtime debug code in every allocating function
``NUMBA_DEBUG_NRT_STACK_LIMIT``     the reference counting calls, under ``NUMBA_DEBUG_NRT``
``NUMBA_FULL_TRACEBACKS``           a runtime message in each bounds check
``NUMBA_DEVELOPER_MODE``            the same, as ``NUMBA_FULL_TRACEBACKS``'s default
``NUMBA_DEBUG_JIT``                 a runtime ``printf`` per lowered instruction
``NUMBA_DEBUG_ARRAY_OPT``           runtime ``printf`` calls in parallel code
``NUMBA_DEBUG_ARRAY_OPT_RUNTIME``   runtime ``printf`` calls in parallel code
``NUMBA_PARFOR_MAX_TUPLE_SIZE``     whether a parallel loop over a tuple compiles
``NUMBA_USE_RVSDG_FRONTEND``        a different frontend (numba 0.60 on Python 3.11 only)
``NUMBA_GDB_BINARY``                the path compiled into a call of ``numba.gdb()``
``NUMBA_DUMP_ASSEMBLY``             register allocation and instruction order
``NUMBA_DEBUG``                     the same, through ``NUMBA_DUMP_ASSEMBLY``
``NUMBA_DIFF_IR``                   code layout
=================================== ===================================================================

numba reads some of them once, when it builds its code generator at the first compile in a process, and others at
every compile, and which is which differs by setting and by version, so the key holds both values for each. numbox
records the first when numba builds its code generator, provided numbox was imported first; where numba compiled
something before numbox was imported, the values at numbox's import stand in, so a setting changed in code between
that first compile and numbox's import is not seen.

Every other setting numba 0.60 to 0.68 reads is in :data:`~numbox.core.cache_key.UNKEYED_SETTINGS` with the reason
it stays out: ``NUMBA_CPU_NAME`` and ``NUMBA_CPU_FEATURES`` are in numba's own key already; ``NUMBA_DISABLE_JIT``
compiles nothing; ``NUMBA_DISABLE_INTEL_SVML`` is left out, SVML being outside what numbox keys; the thread count,
the threading layer and the rest change no code on any version surveyed. numba's parallel code reads the thread
count at run time, and a binary built for one count runs correctly under another; a count frozen into a global at
import is a global like any other, and keyed as one. A test fails on any numba release that reads a ``NUMBA_*``
setting in neither table.

When a function compiles uncached
+++++++++++++++++++++++++++++++++

A value the function reads that has no stable form, one no process could put in the key the same way twice, makes
the function compile without an on-disk cache. It compiles in every process and saves nothing, and an
:class:`~numbox.core.cache_key.UncachedWarning` names the value, once per process. An ``Enum`` member is one: numba
compiles it as a constant, and it is none of the kinds the key puts in by name. Read as a module attribute,
``settings.MODE`` say, it is refused the same way, as is one held in the closure of an ``@intrinsic``, whose code
generator reads it as it compiles. A numba type named after the address of an object, as numba names any type that
holds a jitclass, is refused too. Passing the value as an argument keeps the function cached.

A function the keyed function calls that numba caches under its own key alone, a plain ``numba.njit(cache=True)``
or a ``numba.vectorize(cache=True)`` ufunc, makes it compile uncached as well, as ``compile_kernel`` does for a
kernel. numba loads such a function from its own cache, which a change to a value it reads does not miss, and the
keyed function would link that old code into a binary saved under a key naming the new value. The warning names the
function: decorate it with :func:`~numbox.core.cache_key.keyed_njit`, which keys its own cache too, or drop its
``cache=True``. Until then numba still loads its old code in the process at hand, as it would without numbox.

The churn warning
+++++++++++++++++

A global that takes a new value every run, a timestamp or a random seed, saves a new entry every run, since each value
is a key of its own. Once a function's index holds more than :data:`~numbox.core.cache_key.CHURN_THRESHOLD` entries
(100), the next save draws a :class:`~numbox.core.cache_key.CacheChurnWarning` naming the parts of the key that
differ between the entries, most values first, once per function and process. Nothing is removed. A function
compiled for many signatures holds an entry per signature too, and draws no warning while every part agrees.

Reading the key
+++++++++++++++

:func:`~numbox.core.cache_key.explain_cache_key` lists the parts as they stand for the function's next compile in
this process:

.. code-block:: python

    from numbox.core.cache_key import explain_cache_key, keyed_njit

    SCALE = 2.0


    @keyed_njit(cache=True)
    def scaled(x):
        return x * SCALE


    parts = explain_cache_key(scaled)
    print(parts["settings"]["NUMBA_OPT"])  # ('3', '3'): when numba built its code generator, and now
    print(parts["global __main__.SCALE"])  # the digest of SCALE's value

Two calls that disagree on a part are two different keys, which is how to find what made a function miss.

Upgrading
+++++++++

numbox's key holds parts numba's does not, so a function numba cached under its own key compiles once more, cold,
in each environment, the first time numbox keys it; the entries numba wrote stay in its index and are never loaded
again. numbox's version is in the key, so each upgrade of numbox does the same. An edit to a package's code
generators in place, an ``@intrinsic`` or an ``@overload`` in a development checkout, changes nothing the key can
see: clear the cache after one.

Modules
++++++++

numbox.core.cache_key
---------------------

.. automodule:: numbox.core.cache_key
   :members:
   :show-inheritance:
   :undoc-members:
