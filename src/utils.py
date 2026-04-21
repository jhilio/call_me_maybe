from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Callable, Optional, Tuple
from functools import update_wrapper


class stats_deco(ABC):
    all_monitored: dict[str, dict[str, float]] = {}

    def __init__(self, func: Callable):
        self.func = func
        self._stats: dict[str, float] = {}
        self.all_monitored[f"{func.__module__}.{func.__name__}"] = self._stats
        update_wrapper(self, func)

    def get_stats(self) -> dict[str, float]:
        return self._stats

    @classmethod
    def get_all_stats(cls) -> dict[str, dict[str, float]]:
        return cls.all_monitored

    def __get__(self, instance: Any, owner: Any) -> Callable:
        if instance is None:
            return self

        # bind function to instance like a normal method
        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args: tuple, **kwargs: dict) -> Any:
        pass


class monitor_time(stats_deco):
    def __init__(self, func: Callable):
        super().__init__(func)
        self._stats["total_time"] = 0
        self._stats["call_count"] = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        try:
            result = self.func(*args, **kwargs)
            return result
        finally:
            elapsed = perf_counter() - start
            self._stats["total_time"] = self._stats.get(
                "total_time", 0.0) + elapsed
            self._stats["call_count"] = self._stats.get(
                "call_count", 0) + 1


def format_dict(
        main: dict,
        *keys: Any,
        sort: bool = True,
        sort_key: Optional[Callable[[tuple[Any, Any]], Any]] = None,
        reverse: bool = True,
        value_filter: Optional[Callable[[Any], bool]] = None
        ) -> Tuple[Tuple[Any, Any], ...]:
    result: tuple[tuple[Any, Any], ...] = tuple(
        (key, value)
        for key in keys
        if (value := main.get(key)) is not None
        and (value_filter(value) if callable(value_filter) else True)
    )
    if sort:
        result = tuple(sorted(result, key=sort_key, reverse=reverse))
    return result
