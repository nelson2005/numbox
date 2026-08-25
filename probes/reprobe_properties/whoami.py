import sys
import numbox, numba, llvmlite
print("numbox", numbox.__file__)
print("numba", numba.__version__, "llvmlite", llvmlite.__version__)
print("sys.path0", repr(sys.path[0]))
print("python", sys.version.split()[0])
