import os
import pandas as pd
from functools import wraps


def parquet_cache(filepath):
    """
    Decorator to cache a function's return value as a Parquet file.
    If the file exists, it is read and returned.
    Otherwise, the function is called, its output saved, and returned.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(filepath):
                print(f"Loading cached data from {filepath}")
                return pd.read_parquet(filepath)
            else:
                print(f"Cache not found. Running function and saving to {filepath}")
                result = func(*args, **kwargs)
                result.to_parquet(filepath, index=True)
                return result

        return wrapper

    return decorator
