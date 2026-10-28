import inspect


_REJECTED_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def make_params_strings(func):
    """Render a function's parameters as two source-string fragments for generated ``def`` headers and call sites.

    The function rejects ``*args`` and ``**kwargs`` with ``ValueError``. The existing source-string codegen cannot
    reproduce their ``*`` and ``**`` markers, so the generated wrapper would silently bind those parameters as plain
    positional arguments, a calling convention different from the one the user wrote.

    The function silently flattens keyword-only (``*,``) and positional-only (``/``) markers to positional-or-keyword.
    This is by design: the ``@proxy`` decorator relies on the flattening to support ``Omitted``-style overloads written
    as ``def f(x, *, y=default)``. The proxy wrapper exposes ``f(x, y=default)``, so callers can pass ``y``
    positionally as well. :func:`numbox.core.proxy.proxy` documents this behavior.

    The function formats default-argument values through ``repr(default)``. Primitives whose ``repr()`` is a valid
    Python expression (ints, finite floats, strings, ``None``, booleans, tuples thereof) round-trip. Some values do
    not round-trip. ``float('nan')`` and ``float('inf')`` render as the bare identifiers ``nan`` and ``inf``, which
    raise ``NameError`` at exec time. A complex object whose ``repr()`` is, e.g., ``"<MyClass object at 0x...>"``
    likewise renders to invalid source and raises ``SyntaxError`` or ``NameError``. These are visible failures, not
    silent miscompilations.
    """
    func_params = inspect.signature(func).parameters
    for name, p in func_params.items():
        if p.kind in _REJECTED_KINDS:
            raise ValueError(
                f"{func.__qualname__}: parameter {name!r} of kind "
                f"{_REJECTED_KINDS[p.kind]} is not supported by source-string codegen; "
                f"use explicit positional-or-keyword parameters"
            )
    func_params_str = ', '.join(
        [k if v.default == inspect._empty else f'{k}={v.default!r}' for k, v in func_params.items()]
    )
    func_names_params_str = ', '.join(func_params.keys())
    return func_params_str, func_names_params_str
