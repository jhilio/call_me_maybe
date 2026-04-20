from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Callable
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

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # bind function to instance like a normal method
        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args: tuple, **kwargs: dict) -> Any:
        pass


class monitor_time(stats_deco):
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


# def monitor_time(func: Optional[Callable] = None) -> Callable:
#     if not hasattr(monitor_time, "stats"):
#         monitor_time.stats = {}

#     def decorator(f: Callable) -> Callable:
#         monitor_time.stats[f] = {
#             "total_time": 0.0,
#             "call_count": 0,
#         }

#         @wraps(f)
#         def wrapper(*args: tuple, **kwargs: dict) -> Any:
#             start = perf_counter()
#             try:
#                 return f(*args, **kwargs)
#             finally:
#                 monitor_time.stats[f]["total_time"] += perf_counter() - start
#                 monitor_time.stats[f]["call_count"] += 1
#                 return None

#         return wrapper
#     if func is not None:
#         return decorator(func)
#     return decorator
