"""libc wrappers callable from numba `@njit` code.

See "Bindings module conventions" in ``docs/numbox.core.bindings.rst``
for the ABI-safety, ``@proxy`` caching, and reference-source conventions
shared across all binding modules.
"""
from numbox.core.proxy.proxy import proxy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options


__all__ = [
    "rand", "srand",
    "strlen", "puts", "fputs", "fputc", "putchar",
    "fwrite", "fread", "fflush",
    "fopen", "fclose",
    "feof", "ferror", "clearerr",
    "strcmp", "strncmp", "strchr", "strrchr", "strstr", "strncpy", "strerror",
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    "getenv",
]

load_lib("c")


@proxy(signatures.get("rand"), jit_options=jit_options)
def rand():
    """Return a pseudo-random int in [0, RAND_MAX].

    The C function is not thread-safe.
    See POSIX `rand() <https://man7.org/linux/man-pages/man3/rand.3.html>`_.
    """
    return _call_lib_func("rand", ())


@proxy(signatures.get("srand"), jit_options=jit_options)
def srand(s):
    """Seed the rand() generator.

    See POSIX `srand(seed) <https://man7.org/linux/man-pages/man3/rand.3.html>`_.
    """
    return _call_lib_func("srand", (s,))


@proxy(signatures.get("strlen"), jit_options=jit_options)
def strlen(s):
    """Return the number of bytes in the C string at `s` before the trailing NUL.

    See POSIX `strlen(s) <https://man7.org/linux/man-pages/man3/strlen.3.html>`_.
    """
    return _call_lib_func("strlen", (s,))


@proxy(signatures.get("puts"), jit_options=jit_options)
def puts(s):
    """Write the NUL-terminated string at `s` to stdout, followed by a newline.

    The call returns a non-negative value on success and EOF on error.
    See POSIX `puts(s) <https://man7.org/linux/man-pages/man3/puts.3.html>`_.
    """
    return _call_lib_func("puts", (s,))


@proxy(signatures.get("fputs"), jit_options=jit_options)
def fputs(s, fp):
    """Write the NUL-terminated string at `s` to FILE* `fp`, without a trailing newline.

    The call returns a non-negative value on success and EOF on error.
    See POSIX `fputs(s, fp) <https://man7.org/linux/man-pages/man3/fputs.3.html>`_.
    """
    return _call_lib_func("fputs", (s, fp))


@proxy(signatures.get("fputc"), jit_options=jit_options)
def fputc(c, fp):
    """Write the byte `c`, converted to unsigned char, to FILE* `fp`.

    The call returns the byte on success and EOF on error.
    See POSIX `fputc(c, fp) <https://man7.org/linux/man-pages/man3/fputc.3.html>`_.
    """
    return _call_lib_func("fputc", (c, fp))


@proxy(signatures.get("putchar"), jit_options=jit_options)
def putchar(c):
    """Write the byte `c`, converted to unsigned char, to stdout.

    See POSIX `putchar(c) <https://man7.org/linux/man-pages/man3/putchar.3.html>`_.
    """
    return _call_lib_func("putchar", (c,))


@proxy(signatures.get("fwrite"), jit_options=jit_options)
def fwrite(ptr, size, nmemb, fp):
    """Write `nmemb` elements of `size` bytes each from `ptr` to FILE* `fp`.

    The call returns the number of elements written successfully, which may be less than `nmemb` on error.
    See POSIX `fwrite(ptr, size, nmemb, fp) <https://man7.org/linux/man-pages/man3/fwrite.3.html>`_.
    """
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@proxy(signatures.get("fread"), jit_options=jit_options)
def fread(ptr, size, nmemb, fp):
    """Read up to `nmemb` elements of `size` bytes each from FILE* `fp` into `ptr`.

    The call returns the number of elements read successfully. Use `feof` and `ferror` to determine the cause of a
    short read.
    See POSIX `fread(ptr, size, nmemb, fp) <https://man7.org/linux/man-pages/man3/fread.3.html>`_.
    """
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@proxy(signatures.get("fflush"), jit_options=jit_options)
def fflush(fp):
    """Flush the C stdio buffer for FILE* `fp`, or for all streams if `fp` is NULL (0).

    The call returns 0 on success and EOF on error.
    See POSIX `fflush(fp) <https://man7.org/linux/man-pages/man3/fflush.3.html>`_.
    """
    return _call_lib_func("fflush", (fp,))


@proxy(signatures.get("fopen"), jit_options=jit_options)
def fopen(path, mode):
    """Open the file at the NUL-terminated `path` with the mode string `mode`, for example "r", "wb" or "a+".

    The call returns a FILE* as an intp, or 0 on error, in which case the caller can read errno. The FILE* is an owned
    resource, and the caller must close it with `fclose` to avoid a leak.
    See POSIX `fopen(path, mode) <https://man7.org/linux/man-pages/man3/fopen.3.html>`_.
    """
    return _call_lib_func("fopen", (path, mode))


@proxy(signatures.get("fclose"), jit_options=jit_options)
def fclose(fp):
    """Close FILE* `fp` and flush any buffered output.

    The call returns 0 on success and EOF on error. After the call, `fp` is invalid and must not be reused.
    See POSIX `fclose(fp) <https://man7.org/linux/man-pages/man3/fclose.3.html>`_.
    """
    return _call_lib_func("fclose", (fp,))


@proxy(signatures.get("feof"), jit_options=jit_options)
def feof(fp):
    """Return a non-zero value if and only if the end-of-file indicator is set on FILE* `fp`.

    `clearerr` or a successful repositioning (`fseek` or `rewind`) clears the indicator.
    See POSIX `feof(fp) <https://man7.org/linux/man-pages/man3/feof.3.html>`_.
    """
    return _call_lib_func("feof", (fp,))


@proxy(signatures.get("ferror"), jit_options=jit_options)
def ferror(fp):
    """Return a non-zero value if and only if the error indicator is set on FILE* `fp`.

    Only `clearerr` clears the indicator.
    See POSIX `ferror(fp) <https://man7.org/linux/man-pages/man3/ferror.3.html>`_.
    """
    return _call_lib_func("ferror", (fp,))


@proxy(signatures.get("clearerr"), jit_options=jit_options)
def clearerr(fp):
    """Clear the end-of-file and error indicators on FILE* `fp`.

    See POSIX `clearerr(fp) <https://man7.org/linux/man-pages/man3/clearerr.3.html>`_.
    """
    return _call_lib_func("clearerr", (fp,))


@proxy(signatures.get("strcmp"), jit_options=jit_options)
def strcmp(a, b):
    """Compare two NUL-terminated strings lexicographically, byte by byte.

    The call returns a negative value if a<b, zero if the strings are equal, and a positive value if a>b.
    See POSIX `strcmp(a, b) <https://man7.org/linux/man-pages/man3/strcmp.3.html>`_.
    """
    return _call_lib_func("strcmp", (a, b))


@proxy(signatures.get("strncmp"), jit_options=jit_options)
def strncmp(a, b, n):
    """Compare at most the first `n` bytes of two NUL-terminated strings, as `strcmp` does.

    The comparison stops early at a NUL in either string. With `n == 0`, the call always returns 0, as the
    specification requires.
    See POSIX `strncmp(a, b, n) <https://man7.org/linux/man-pages/man3/strncmp.3.html>`_.
    """
    return _call_lib_func("strncmp", (a, b, n))


@proxy(signatures.get("strchr"), jit_options=jit_options)
def strchr(s, c):
    """Return a pointer to the first occurrence of the byte `c` in the NUL-terminated string `s`, or 0 if not found.

    With `c==0`, the call returns a pointer to the terminating NUL.
    See POSIX `strchr(s, c) <https://man7.org/linux/man-pages/man3/strchr.3.html>`_.
    """
    return _call_lib_func("strchr", (s, c))


@proxy(signatures.get("strrchr"), jit_options=jit_options)
def strrchr(s, c):
    """Return a pointer to the last occurrence of the byte `c` in the NUL-terminated string `s`, or 0 if not found.

    See POSIX `strrchr(s, c) <https://man7.org/linux/man-pages/man3/strrchr.3.html>`_.
    """
    return _call_lib_func("strrchr", (s, c))


@proxy(signatures.get("strstr"), jit_options=jit_options)
def strstr(haystack, needle):
    """Return a pointer to the first occurrence of the NUL-terminated `needle` within the NUL-terminated `haystack`.

    The call returns 0 if `needle` is not found, and `haystack` if `needle` is empty.
    See POSIX `strstr(haystack, needle) <https://man7.org/linux/man-pages/man3/strstr.3.html>`_.
    """
    return _call_lib_func("strstr", (haystack, needle))


@proxy(signatures.get("strncpy"), jit_options=jit_options)
def strncpy(dst, src, n):
    """Copy at most n bytes from src to dst.

    strncpy does not guarantee null termination: if strlen(src) >= n, dst contains n bytes from src and no trailing
    NUL. A caller that needs a NUL-terminated result must reserve an extra byte. It must then either zero the buffer
    before the call or write dst[n] = 0 explicitly after the call.
    See POSIX `strncpy(dst, src, n) <https://man7.org/linux/man-pages/man3/strncpy.3.html>`_.
    """
    return _call_lib_func("strncpy", (dst, src, n))


@proxy(signatures.get("strerror"), jit_options=jit_options)
def strerror(errnum):
    """Return a pointer to the static error-message string for errnum.

    strerror is not thread-safe. The returned pointer references a per-process static buffer, which subsequent
    strerror calls may overwrite. Use ``strerror_safe`` (in ``numbox.core.bindings.strerror``) for thread-safe
    operation.
    See POSIX `strerror(errnum) <https://man7.org/linux/man-pages/man3/strerror.3.html>`_.
    """
    return _call_lib_func("strerror", (errnum,))


@proxy(signatures.get("memcpy"), jit_options=jit_options)
def memcpy(dst, src, n):
    """Copy `n` bytes from `src` to `dst`.

    The source and destination must not overlap; use `memmove` for overlapping regions. The call returns `dst`.
    See POSIX `memcpy(dst, src, n) <https://man7.org/linux/man-pages/man3/memcpy.3.html>`_.
    """
    return _call_lib_func("memcpy", (dst, src, n))


@proxy(signatures.get("memmove"), jit_options=jit_options)
def memmove(dst, src, n):
    """Copy `n` bytes from `src` to `dst`, with correct results if the two regions overlap.

    The call returns `dst`.
    See POSIX `memmove(dst, src, n) <https://man7.org/linux/man-pages/man3/memmove.3.html>`_.
    """
    return _call_lib_func("memmove", (dst, src, n))


@proxy(signatures.get("memset"), jit_options=jit_options)
def memset(dst, c, n):
    """Fill the first `n` bytes of `dst` with the byte value `c & 0xff`.

    The call returns `dst`.
    See POSIX `memset(dst, c, n) <https://man7.org/linux/man-pages/man3/memset.3.html>`_.
    """
    return _call_lib_func("memset", (dst, c, n))


@proxy(signatures.get("memcmp"), jit_options=jit_options)
def memcmp(a, b, n):
    """Compare the first `n` bytes of `a` and `b`, byte by byte.

    The call returns a negative, zero or positive value, with the same sign convention as `strcmp`. Unlike `strcmp`,
    it compares raw bytes and does not stop at a NUL.
    See POSIX `memcmp(a, b, n) <https://man7.org/linux/man-pages/man3/memcmp.3.html>`_.
    """
    return _call_lib_func("memcmp", (a, b, n))


@proxy(signatures.get("memchr"), jit_options=jit_options)
def memchr(s, c, n):
    """Return a pointer to the first byte equal to `c & 0xff` in the first `n` bytes at `s`, or 0 if not found.

    See POSIX `memchr(s, c, n) <https://man7.org/linux/man-pages/man3/memchr.3.html>`_.
    """
    return _call_lib_func("memchr", (s, c, n))


@proxy(signatures.get("getenv"), jit_options=jit_options)
def getenv(name):
    """Return a pointer to the value string of the variable `name` in the process environ table, or 0 if unset.

    The platform environ owns the returned pointer. Do not mutate or free it, and do not assume that it remains valid
    after a subsequent setenv or putenv. A caller that needs a stable Python str should copy the value with
    `get_str_from_p_as_int` before mutating environ.
    See POSIX `getenv(name) <https://man7.org/linux/man-pages/man3/getenv.3.html>`_.
    """
    return _call_lib_func("getenv", (name,))
