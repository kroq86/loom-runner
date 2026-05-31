from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCategory(StrEnum):
    TRANSIENT = "transient"
    VALIDATION = "validation"
    BUSINESS = "business"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    retryable_categories: frozenset[ErrorCategory] = field(
        default_factory=lambda: frozenset({ErrorCategory.TRANSIENT})
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        object.__setattr__(
            self,
            "retryable_categories",
            frozenset(_coerce_category(category) for category in self.retryable_categories),
        )

    def should_retry(self, category: ErrorCategory | str, attempt_index: int) -> bool:
        return (
            _coerce_category(category) in self.retryable_categories
            and attempt_index + 1 < self.max_attempts
        )


class StepExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_category: ErrorCategory | str = ErrorCategory.UNKNOWN,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.error_category = _coerce_category(error_category)
        self.is_retryable = (
            self.error_category == ErrorCategory.TRANSIENT
            if is_retryable is None
            else is_retryable
        )


def error_category_for(exc: BaseException) -> ErrorCategory:
    category = getattr(exc, "error_category", ErrorCategory.UNKNOWN)
    return _coerce_category(category)


def is_retryable_error(exc: BaseException) -> bool:
    retryable = getattr(exc, "is_retryable", None)
    if retryable is not None:
        return bool(retryable)
    return error_category_for(exc) == ErrorCategory.TRANSIENT


def _coerce_category(category: ErrorCategory | str) -> ErrorCategory:
    if isinstance(category, ErrorCategory):
        return category
    try:
        return ErrorCategory(str(category))
    except ValueError:
        return ErrorCategory.UNKNOWN
