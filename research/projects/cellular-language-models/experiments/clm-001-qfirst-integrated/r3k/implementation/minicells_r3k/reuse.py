"""Bind existing stateless R3j operations to R3k identity/storage, without global mutation."""

from types import FunctionType


def bind(function, **dependencies):
    namespace = dict(function.__globals__)
    namespace.update(dependencies)
    result = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result
