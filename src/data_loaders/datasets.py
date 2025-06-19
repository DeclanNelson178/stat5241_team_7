from enum import Enum
from src.data_loaders.rollcall import get_training_data_v1, get_trainig_data_v2


class Datasets(Enum):
    V1 = "v1"
    """Basic bill info one hot encoding, vote type, senator id"""

    V2 = "v2"

    def get_dataset(self):
        return {
            Datasets.V1: get_training_data_v1,
            Datasets.V2: get_trainig_data_v2,
        }[self]()
