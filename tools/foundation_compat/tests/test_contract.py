# Contract tests of the foundation backends (level 1)
# ==============================================================================
# Check that the installed backend still exposes the symbols and arguments the
# adapter uses. No weights are downloaded.
import importlib
import inspect
from contracts import BACKEND_CONTRACTS


def resolve_target(target):
    """
    Import the object a `'module:qualified.name'` target refers to.

    Returns
    -------
    obj : object
        The class, method or attribute.
    owner : object
        The object `obj` was taken from (the module for a top-level name).

    """
    module_name, qualified_name = target.split(":")
    owner = None
    obj = importlib.import_module(module_name)
    for name in qualified_name.split("."):
        owner = obj
        obj = getattr(owner, name)

    return obj, owner


def test_backend_has_contract(backend):
    """
    Test that the adapter selected with `--backend` declares its backend calls
    in `BACKEND_CONTRACTS`.
    """
    assert BACKEND_CONTRACTS.get(backend), (
        f"'{backend}' has no entry in BACKEND_CONTRACTS "
        f"(tools/foundation_compat/tests/contracts.py)."
    )


def test_backend_call_matches_signature(contract_call):
    """
    Test that the target of a backend call exists and that its signature
    accepts the arguments the adapter passes to it.
    """
    target = contract_call["target"]
    kwargs = contract_call.get("kwargs", [])
    weak_kwargs = contract_call.get("weak_kwargs", [])
    n_positional = contract_call.get("n_positional", 0)

    obj, owner = resolve_target(target)

    if not (kwargs or weak_kwargs or n_positional):
        return

    signature = inspect.signature(obj)
    parameters = signature.parameters
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    # A plain function taken from its class still expects `self`.
    if inspect.isclass(owner) and inspect.isfunction(obj):
        n_positional += 1

    try:
        signature.bind_partial(
            *[None] * n_positional,
            **{name: None for name in (*kwargs, *weak_kwargs)},
        )
    except TypeError as exc:
        raise AssertionError(
            f"`{target}` does not accept the call of the adapter: {exc}. "
            f"Call: {n_positional} positional argument(s) and the keyword "
            f"arguments {[*kwargs, *weak_kwargs]}. Signature: {signature}."
        ) from exc

    # `bind_partial` accepts any name when the signature has `**kwargs`, so
    # the arguments are also required to be declared explicitly.
    not_declared = [
        name for name in kwargs
        if name not in parameters
        or parameters[name].kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert not not_declared, (
        f"`{target}{signature}` does not declare the arguments {not_declared} "
        f"that the adapter passes by name."
    )

    if weak_kwargs:
        assert accepts_var_keyword or all(name in parameters for name in weak_kwargs)
