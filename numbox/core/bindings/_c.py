from numbox.utils.highlevel import cres
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


@cres(signatures.get("puts"), cache=True)
def puts(s):
    return _call_lib_func("puts", (s,))


@cres(signatures.get("fputs"), cache=True)
def fputs(s, fp):
    return _call_lib_func("fputs", (s, fp))


@cres(signatures.get("fputc"), cache=True)
def fputc(c, fp):
    return _call_lib_func("fputc", (c, fp))


@cres(signatures.get("putchar"), cache=True)
def putchar(c):
    return _call_lib_func("putchar", (c,))


@cres(signatures.get("fwrite"), cache=True)
def fwrite(ptr, size, nmemb, fp):
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@cres(signatures.get("fread"), cache=True)
def fread(ptr, size, nmemb, fp):
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@cres(signatures.get("fflush"), cache=True)
def fflush(fp):
    return _call_lib_func("fflush", (fp,))


@cres(signatures.get("fopen"), cache=True)
def fopen(path, mode):
    return _call_lib_func("fopen", (path, mode))


@cres(signatures.get("fclose"), cache=True)
def fclose(fp):
    return _call_lib_func("fclose", (fp,))


@cres(signatures.get("feof"), cache=True)
def feof(fp):
    return _call_lib_func("feof", (fp,))


@cres(signatures.get("ferror"), cache=True)
def ferror(fp):
    return _call_lib_func("ferror", (fp,))


@cres(signatures.get("clearerr"), cache=True)
def clearerr(fp):
    return _call_lib_func("clearerr", (fp,))
