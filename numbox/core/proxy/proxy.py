import inspect
from llvmlite import ir  # noqa: F401
from numba import njit
from numba.core import cgutils  # noqa: F401
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.extending import intrinsic  # noqa: F401
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.utils.standard import make_params_strings


def make_proxy_name(name):
    return f'__{name}'


def proxy(sig, jit_options: Optional[dict] = None):
    """ Create a proxy for the decorated function `func` with the given signature(s) `sig`.

    The original function `func` will be eagerly JIT-compiled with the given signature(s).
    A proxy with the name `func_proxy_name` will be created to call `func` in the LLVM scope.
    The original function's variable will be bound to the proxy, i.e., calling the decorated
    function will call the proxy.

    The proxy is a JIT-compiled wrap that invokes the intrinsic that *declares* the `func`
    and calls it with the original arguments. Declaration instructions are relatively cheap
    to statically link into (potential) caller's LLVM code, which is the main motivation behind
    this decorator.

    Machine code for `func` can be cached when so specified in `jit_options`, in which case its
    JIT-compilation will load the `func` into the LLVM scope. Caching option is the other major
    motivation for this decorator, without the need to cache one can avoid static linking
    of the callee's LLVM code into the caller's by simply ignoring the former.

    In case when more than one signature is provided as the `sig` parameter, it is assumed
    that the first signature is the 'main' one while the other ones are supplied to
    allow for the `Omitted` types with default values for (some of) the parameters.

    The returned dispatcher also exposes ``.as_func``: a ``CompileResultWAP``
    (numba ``FunctionType`` value) for the main signature. Use it when a binding
    must be passed as a callback argument or stored in a struct field — the
    dispatcher itself is the call-site form. Referencing ``.as_func`` as a
    Python global from ``@njit(cache=True)`` triggers
    ``lower_constant_function_type`` → ``add_dynamic_addr`` and disables that
    caller's cache, same as plain ``@cres``; call the dispatcher directly for
    cacheable usage.

    See tests for some examples of the use cases.
    """
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    jit_opts.update(jit_opts, inline='always')

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        func_jit = njit(sig, **jit_options)(func)
        llvm_cfunc_wrapper_name = func_jit.get_compile_result(main_sig).fndesc.llvm_cfunc_wrapper_name
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        code_txt = f"""
@intrinsic
def _{func_proxy_name}(typingctx, {func_names_args_str}):
    def codegen(context, builder, signature, args):
        func_ty_ll = ir.FunctionType(
            context.get_data_type(main_sig.return_type),
            [context.get_data_type(arg) for arg in main_sig.args]
        )
        f = cgutils.get_or_insert_function(builder.module, func_ty_ll, "{llvm_cfunc_wrapper_name}")
        return builder.call(f, args)
    return main_sig, codegen

@njit(sig, **jit_opts)
def {func_proxy_name}({func_args_str}):
    return _{func_proxy_name}({func_names_args_str})
"""
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': njit,
                'sig': sig, 'main_sig': main_sig
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        # Anchor the exec'd wrapper at the original decorated function's source
        # line in the user's .py file. Prepended blank lines push the wrapper's
        # ``@njit`` decorator (which is what Python records as the wrapper's
        # ``co_firstlineno``) to ``func.__code__.co_firstlineno`` — the user's
        # ``@proxy`` decorator line. ``inspect.findsource(wrapper)`` then
        # matches that ``@proxy`` line on its very first check (the regex
        # ``r'^(\s*@)'``) and tokenization starts from real, syntactically
        # valid Python at that line. This sidesteps CPython #122981 (a
        # co_firstlineno landing inside a multi-line docstring used to raise
        # tokenize.TokenError) AND avoids a subtler hazard: ``findsource``
        # scans backward from co_firstlineno looking for any line that starts
        # with ``@``, and a docstring containing the text ``@njit(...)`` or
        # similar would be matched as if it were a real decorator, with the
        # tokenizer then trying to parse the docstring content as code.
        # numba's source stamp for a file-backed cached @njit is
        # ``(st_mtime, st_size)`` of co_filename, so user edits to the file
        # correctly invalidate the cache. Stale-cache risk exists only when
        # this template itself changes without a corresponding edit to the
        # user file — treated as developer-managed (clear ``~/.cache/numba/``
        # when shipping a proxy wrapper-template change).
        code_lines = code_txt.split('\n')
        njit_lineno_in_txt = next(
            i + 1 for i, line in enumerate(code_lines) if line.startswith('@njit(')
        )
        prepend = max(0, func.__code__.co_firstlineno - njit_lineno_in_txt)
        prefixed = '\n' * prepend + code_txt
        code = compile(prefixed, inspect.getfile(func), mode='exec')
        exec(code, ns)
        dispatcher = ns[func_proxy_name]
        dispatcher.as_func = CompileResultWAP(func_jit.get_compile_result(main_sig))
        return dispatcher
    return wrap
