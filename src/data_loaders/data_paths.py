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


def get_model_root():
    return get_repo_root() / "data" / "model"


DATA_VERSION = 11  # increment caches to re-run everything
ROLLCALL_QUERY = get_data_root() / f"rollcall_query_{str(DATA_VERSION)}.parquet"
ROLLCALL_CLEANED = get_data_root() / f"rollcall_cleaned_{str(DATA_VERSION)}.parquet"
ROLLCALL_CRS_POLICY = (
    get_data_root() / f"rollcall_crs_policy_{str(DATA_VERSION)}.parquet"
)
RAW_INDIVIDUAL_VOTES = get_data_root() / "house_senate_votes.parquet"
INDIVIDUAL_VOTES_V1 = get_data_root() / f"individual_votes_{str(DATA_VERSION)}.parquet"
TRAINING_DATA_V1 = get_data_root() / f"training_data_v1_{str(DATA_VERSION)}.parquet"

RAW_PARTY_MEMBERSHIP = get_data_root() / "house_senate_members.csv"
VOTE_WITH_PARTY = (
    get_data_root() / f"individual_votes_with_party_{str(DATA_VERSION)}.csv"
)
VOTE_WITH_PARTY_ENRICHED = (
    get_data_root() / f"individual_votes_with_party_enriched_{str(DATA_VERSION)}.csv"
)
TRAINING_DATA_V2 = get_data_root() / f"training_data_v2_{str(DATA_VERSION)}.csv"
