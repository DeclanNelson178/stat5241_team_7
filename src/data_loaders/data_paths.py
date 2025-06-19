from pathlib import Path
import os


def get_repo_root() -> Path:
    """
    Walks up from the current directory to find the root of the git repository.
    Returns the absolute path to the repo root.
    """
    start_path = os.getcwd()
    current = os.path.abspath(start_path)

    while current != os.path.dirname(current):  # stop at filesystem root
        if os.path.isdir(os.path.join(current, ".git")):
            return Path(current) / "src"
        current = os.path.dirname(current)

    raise FileNotFoundError(
        "No .git directory found — are you inside a Git repository?"
    )


def get_data_root():
    return get_repo_root() / "data"


DATA_VERSION = 5

ROLLCALL_QUERY = get_data_root() / f"rollcall_query_{str(DATA_VERSION)}.parquet"
ROLLCALL_CLEANED = get_data_root() / f"rollcall_cleaned_{str(DATA_VERSION)}.parquet"
ROLLCALL_CRS_POLICY = (
    get_data_root() / f"rollcall_crs_policy_{str(DATA_VERSION)}.parquet"
)
