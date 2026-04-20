from time import perf_counter
from functools import wraps

_all_stats = {}

def monitor_time(func=None, *, show=False):
    if show:
        return _all_stats

    def decorator(f):
        _all_stats[f] = {
            "total_time": 0.0,
            "call_count": 0,
        }

        @wraps(f)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return f(*args, **kwargs)
            finally:
                stats = _all_stats[f]
                stats["total_time"] += perf_counter() - start
                stats["call_count"] += 1

        return wrapper
    if func is not None:
        return decorator(func)
    return decorator