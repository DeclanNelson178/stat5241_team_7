import functools
import os
import pandas as pd
from functools import wraps
from src.data_loaders.data_paths import DATA_VERSION, get_data_root, get_model_root


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


def parquet_cache_for_model():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            base_path = get_model_root()
            os.makedirs(base_path, exist_ok=True)
            file_name = (
                f"{self.model_name}_{self.model_identifier}_{DATA_VERSION}.parquet"
            )
            path = os.path.join(base_path, file_name)
            if os.path.exists(path):
                print(f"[Cache] Loaded results from {path}")
                return pd.read_parquet(path)
            result = func(self, *args, **kwargs)
            result.to_parquet(path)
            print(f"[Cache] Saved results to {path}")
            return result

        return wrapper

    return decorator
