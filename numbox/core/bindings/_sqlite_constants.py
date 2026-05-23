"""SQLite numeric constants (result codes, type codes, open flags, blob flags,
trace flags, destructor sentinels).

Public surface — imported via star-import by ``numbox/core/bindings/__init__.py``.
All names are uppercase ``SQLITE_*`` to avoid collision with the lowercase
C-function-named wrappers.

Numba handles Python integer literals natively in ``@njit`` code, so these
constants are usable inside JITed functions without further wrapping. The
underlying SQLite values are API-stable across all matrix versions
(3.34.0 through current).
"""

# === Primary result codes (sqlite3.h) ===
SQLITE_OK = 0
SQLITE_ERROR = 1
SQLITE_INTERNAL = 2
SQLITE_PERM = 3
SQLITE_ABORT = 4
SQLITE_BUSY = 5
SQLITE_LOCKED = 6
SQLITE_NOMEM = 7
SQLITE_READONLY = 8
SQLITE_INTERRUPT = 9
SQLITE_IOERR = 10
SQLITE_CORRUPT = 11
SQLITE_NOTFOUND = 12
SQLITE_FULL = 13
SQLITE_CANTOPEN = 14
SQLITE_PROTOCOL = 15
SQLITE_EMPTY = 16
SQLITE_SCHEMA = 17
SQLITE_TOOBIG = 18
SQLITE_CONSTRAINT = 19
SQLITE_MISMATCH = 20
SQLITE_MISUSE = 21
SQLITE_NOLFS = 22
SQLITE_AUTH = 23
SQLITE_FORMAT = 24
SQLITE_RANGE = 25
SQLITE_NOTADB = 26
SQLITE_NOTICE = 27
SQLITE_WARNING = 28
SQLITE_ROW = 100
SQLITE_DONE = 101

# === Column type codes (sqlite3_column_type return values) ===
SQLITE_INTEGER = 1
SQLITE_FLOAT = 2
SQLITE_TEXT = 3
SQLITE_BLOB = 4
SQLITE_NULL = 5

# === sqlite3_open_v2 flags (combinable with bitwise OR) ===
SQLITE_OPEN_READONLY = 0x00000001
SQLITE_OPEN_READWRITE = 0x00000002
SQLITE_OPEN_CREATE = 0x00000004
SQLITE_OPEN_URI = 0x00000040
SQLITE_OPEN_MEMORY = 0x00000080
SQLITE_OPEN_NOMUTEX = 0x00008000
SQLITE_OPEN_FULLMUTEX = 0x00010000
SQLITE_OPEN_SHAREDCACHE = 0x00020000
SQLITE_OPEN_PRIVATECACHE = 0x00040000

# === sqlite3_blob_open flags (the integer values its `flags` arg accepts) ===
SQLITE_BLOB_READONLY = 0
SQLITE_BLOB_READWRITE = 1

# === sqlite3_trace_v2 event mask bits ===
SQLITE_TRACE_STMT = 0x01
SQLITE_TRACE_PROFILE = 0x02
SQLITE_TRACE_ROW = 0x04
SQLITE_TRACE_CLOSE = 0x08

# === Destructor sentinels for sqlite3_bind_text / sqlite3_bind_blob ===
SQLITE_STATIC = 0
SQLITE_TRANSIENT = -1
