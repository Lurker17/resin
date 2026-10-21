# FlashInfer 0.6.15.post1 annotates this decorator as bare Callable, which erases
# the signatures of constructors and plan() methods. Preserve the wrapped API.
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

_P = ParamSpec("_P")
_R = TypeVar("_R")

@overload
def flashinfer_api(func: Callable[_P, _R], *, trace: Any = ...) -> Callable[_P, _R]: ...
@overload
def flashinfer_api(
    func: None = ..., *, trace: Any = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
