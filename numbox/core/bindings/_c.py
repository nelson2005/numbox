from numbox.utils.highlevel import cres
from numbox.core.bindings._cacheable import cres_cacheable
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib


load_lib("c")


@cres(signatures.get("rand"), cache=True)
def rand():
    return _call_lib_func("rand", ())


@cres(signatures.get("srand"), cache=True)
def srand(s):
    return _call_lib_func("srand", (s,))


@cres(signatures.get("strlen"), cache=True)
def strlen(s):
    return _call_lib_func("strlen", (s,))


@cres_cacheable(signatures.get("puts"), cache=True)
def puts(s):
    return _call_lib_func("puts", (s,))


@cres_cacheable(signatures.get("fputs"), cache=True)
def fputs(s, fp):
    return _call_lib_func("fputs", (s, fp))


@cres_cacheable(signatures.get("fputc"), cache=True)
def fputc(c, fp):
    return _call_lib_func("fputc", (c, fp))


@cres_cacheable(signatures.get("putchar"), cache=True)
def putchar(c):
    return _call_lib_func("putchar", (c,))


@cres_cacheable(signatures.get("fwrite"), cache=True)
def fwrite(ptr, size, nmemb, fp):
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@cres_cacheable(signatures.get("fread"), cache=True)
def fread(ptr, size, nmemb, fp):
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@cres_cacheable(signatures.get("fflush"), cache=True)
def fflush(fp):
    return _call_lib_func("fflush", (fp,))


@cres_cacheable(signatures.get("fopen"), cache=True)
def fopen(path, mode):
    return _call_lib_func("fopen", (path, mode))


@cres_cacheable(signatures.get("fclose"), cache=True)
def fclose(fp):
    return _call_lib_func("fclose", (fp,))


@cres_cacheable(signatures.get("feof"), cache=True)
def feof(fp):
    return _call_lib_func("feof", (fp,))


@cres_cacheable(signatures.get("ferror"), cache=True)
def ferror(fp):
    return _call_lib_func("ferror", (fp,))


@cres_cacheable(signatures.get("clearerr"), cache=True)
def clearerr(fp):
    return _call_lib_func("clearerr", (fp,))


@cres_cacheable(signatures.get("strcmp"), cache=True)
def strcmp(a, b):
    return _call_lib_func("strcmp", (a, b))


@cres_cacheable(signatures.get("strncmp"), cache=True)
def strncmp(a, b, n):
    return _call_lib_func("strncmp", (a, b, n))


@cres_cacheable(signatures.get("strchr"), cache=True)
def strchr(s, c):
    return _call_lib_func("strchr", (s, c))


@cres_cacheable(signatures.get("strrchr"), cache=True)
def strrchr(s, c):
    return _call_lib_func("strrchr", (s, c))


@cres_cacheable(signatures.get("strstr"), cache=True)
def strstr(haystack, needle):
    return _call_lib_func("strstr", (haystack, needle))


@cres_cacheable(signatures.get("strncpy"), cache=True)
def strncpy(dst, src, n):
    """Copy at most n bytes from src to dst (POSIX strncpy semantics).

    Does NOT guarantee null termination: if strlen(src) >= n, dst will
    contain n bytes from src with no trailing NUL. Callers that need a
    NUL-terminated result must reserve an extra byte and either pre-zero
    the buffer or explicitly write dst[n] = 0 after the call.
    """
    return _call_lib_func("strncpy", (dst, src, n))


@cres_cacheable(signatures.get("strerror"), cache=True)
def strerror(errnum):
    """Return a pointer to the static error-message string for errnum.

    NOT thread-safe — the returned pointer references a per-process
    static buffer that subsequent strerror calls may overwrite. Use
    strerror_safe for thread-safe operation.
    """
    return _call_lib_func("strerror", (errnum,))


@cres_cacheable(signatures.get("memcpy"), cache=True)
def memcpy(dst, src, n):
    return _call_lib_func("memcpy", (dst, src, n))


@cres_cacheable(signatures.get("memmove"), cache=True)
def memmove(dst, src, n):
    return _call_lib_func("memmove", (dst, src, n))


@cres_cacheable(signatures.get("memset"), cache=True)
def memset(dst, c, n):
    return _call_lib_func("memset", (dst, c, n))


@cres_cacheable(signatures.get("memcmp"), cache=True)
def memcmp(a, b, n):
    return _call_lib_func("memcmp", (a, b, n))


@cres_cacheable(signatures.get("memchr"), cache=True)
def memchr(s, c, n):
    return _call_lib_func("memchr", (s, c, n))


@cres_cacheable(signatures.get("getenv"), cache=True)
def getenv(name):
    """Return pointer to the value string in the process environ table.

    The returned pointer is owned by the platform environ — do NOT
    mutate, free, or assume it survives a subsequent setenv/putenv.
    Callers that need a stable Python str should copy via
    `get_str_from_p_as_int` before mutating environ.
    """
    return _call_lib_func("getenv", (name,))
