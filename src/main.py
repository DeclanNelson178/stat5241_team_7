"""You can run this file using python -m src.main from the root directory"""

from src.data_loaders.rollcall import (
    get_cleaned_rollcall_data,
    get_individual_votes,
    get_training_data_v1,
)
from src.models.helpers import split_data
from src.models.models import RegisteredModels

if __name__ == "__main__":
    eval = RegisteredModels.model_registry()[-1].run_model()
    print(eval)
