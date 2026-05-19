"""
Reusable Decorators
===================
Production-quality decorators for timing, retrying, and validating
DataFrame inputs across pipeline and analysis modules.

Usage:
    from utils.decorators import timer, retry, validate_dataframe

    @timer
    def heavy_computation():
        ...

    @retry(max_attempts=3, delay=1.0)
    def flaky_db_call():
        ...
"""

import time
import logging
import functools
from typing import Optional, List, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# @timer
# ────────────────────────────────────────────────

def timer(func: Callable = None, *, log_level: str = "INFO"):
    """
    Log the execution duration of a function.

    Can be used bare (``@timer``) or with arguments (``@timer(log_level="DEBUG")``).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start

            level = getattr(logging, log_level.upper(), logging.INFO)
            logger.log(level, f"{fn.__qualname__} completed in {elapsed:.3f}s")
            return result
        return wrapper

    if func is not None:
        # Called as @timer (no parentheses)
        return decorator(func)
    # Called as @timer(log_level="DEBUG")
    return decorator


# ────────────────────────────────────────────────
# @retry
# ────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Retry a function with exponential backoff on specified exceptions.

    Args:
        max_attempts: Maximum number of tries.
        delay:        Initial delay between retries (seconds).
        backoff:      Multiplier applied to delay after each failure.
        exceptions:   Tuple of exception classes to catch.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{fn.__qualname__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{fn.__qualname__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


# ────────────────────────────────────────────────
# @validate_dataframe
# ────────────────────────────────────────────────

def validate_dataframe(
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
    arg_name: str = "df",
):
    """
    Validate that the DataFrame argument meets schema requirements.

    Args:
        required_columns: List of column names that must be present.
        min_rows:         Minimum number of rows expected.
        arg_name:         Name of the DataFrame parameter in the function signature.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Resolve the DataFrame from positional or keyword args
            import inspect
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())

            if arg_name in kwargs:
                df = kwargs[arg_name]
            elif arg_name in params:
                idx = params.index(arg_name)
                if idx < len(args):
                    df = args[idx]
                else:
                    raise ValueError(f"Missing required argument: {arg_name}")
            else:
                raise ValueError(f"Function has no parameter named '{arg_name}'")

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for '{arg_name}', got {type(df).__name__}")

            if len(df) < min_rows:
                raise ValueError(
                    f"DataFrame '{arg_name}' has {len(df)} rows, "
                    f"but {min_rows} required"
                )

            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"DataFrame '{arg_name}' missing columns: {missing}"
                    )

            return fn(*args, **kwargs)
        return wrapper
    return decorator
