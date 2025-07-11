"""You can run this file using python -m src.main from the root directory"""

from src.data_loaders.rollcall import (
    get_cleaned_rollcall_data,
    get_individual_votes,
    get_training_data_v1,
)
from src.models.helpers import split_data
from src.models.models import RegisteredModels
from src.data_loaders.data_paths import get_data_root

if __name__ == "__main__":
    eval = RegisteredModels.run_models()
    eval.to_csv(get_data_root() / "evaluation_full_2.csv")
