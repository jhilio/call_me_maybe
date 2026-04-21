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
        """bind self to methode

        Args:
            instance (Any):
            instance of the class that \
            the attribute is accessed on (or None if accessed via the class)
            owner (Any):
                the class that defines the descriptor

        Returns:
            Callable:
            a bound callable where the instance \
                is implicitly passed as the first argument
        """
        if instance is None:
            return self

        # bind function to instance like a normal method
        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args: tuple, **kwargs: dict) -> Any:
        pass


class monitor_time(stats_deco):
    def __init__(self, func: Callable):
        """init the decorator

        Args:
            func (Callable): function to bounc to
        """
        super().__init__(func)
        self._stats["total_time"] = 0
        self._stats["call_count"] = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """call decored function and add time spent

        Returns:
            Any: function result
        """
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
    """format specified entry of a dict as a tuple of \
    (key, value) with optional filter and sort

    Args:
        main (dict):
            source dictionary to extract values from
        *keys (Any):
            keys to extract from the dictionary

        sort (bool, optional):
            whether to sort the resulting tuples. Defaults to True.
        sort_key (Optional[Callable[[tuple[Any, Any]], Any]], optional):
            function used to compute sorting key for each (key, value) pair. \
        Defaults to None.
        reverse (bool, optional):
            whether to sort in descending order. \
        Defaults to True.
        value_filter (Optional[Callable[[Any], bool]], optional):
            function that filters values; \
        only values returning True are included. \
        Defaults to None.

    Returns:
        Tuple[Tuple[Any, Any], ...]:
            tuple of (key, value) pairs extracted from the dictionary, \
        optionally filtered and sorted
    """
    result: tuple[tuple[Any, Any], ...] = tuple(
        (key, value)
        for key in keys
        if (value := main.get(key)) is not None
        and (value_filter(value) if callable(value_filter) else True)
    )
    if sort:
        result = tuple(sorted(result, key=sort_key, reverse=reverse))
    return result
