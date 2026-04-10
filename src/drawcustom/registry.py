from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from .types import DrawingContext, ElementType

# Type alias for element handler functions
_HandlerFn = Callable[..., Coroutine[Any, Any, None]]

# Global registry populated by decorators
_handlers: dict[ElementType, tuple[_HandlerFn, list[str]]] = {}


def element_handler(element_type: ElementType, requires: list[str] | None = None) -> Callable[[_HandlerFn], _HandlerFn]:
    """
    Decorator to register and validate element handlers.

    Args:
        element_type: The ElementType this handler processes
        requires: List of required element keys (validated before handler runs)
    """

    def decorator(func: _HandlerFn) -> _HandlerFn:
        @wraps(func)
        async def wrapper(ctx: DrawingContext, element: dict[str, Any]) -> None:
            if requires:
                missing = [key for key in requires if key not in element]
                if missing:
                    raise ValueError(f"{element_type.value} requires: {', '.join(missing)}")
            await func(ctx, element)

        _handlers[element_type] = (wrapper, requires or [])
        return wrapper

    return decorator


def get_all_handlers() -> dict[ElementType, tuple[_HandlerFn, list[str]]]:
    """Return all registered handlers."""
    return _handlers
