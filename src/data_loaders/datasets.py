from enum import Enum
from src.data_loaders.rollcall import (
    get_trainig_data_v3,
    get_trainig_data_v4,
    get_training_data_v1,
    get_trainig_data_v2,
    get_training_data_v5,
)


class Datasets(Enum):
    V1 = "v1"
    """Basic bill info one hot encoding, vote type, senator id"""

    V2 = "v2"
    """V1 + party affiliation"""

    V3 = "v3"
    """V2 + population, terms served, prior vote history on the same bill"""

    V4 = "v4"
    """V3 + state code and district code """

    V5 = "v5"
    """V4 + lobbyist data"""

    def get_dataset(self):
        return {
            Datasets.V1: get_training_data_v1,
            Datasets.V2: get_trainig_data_v2,
            Datasets.V3: get_trainig_data_v3,
            Datasets.V4: get_trainig_data_v4,
            Datasets.V5: get_training_data_v5,
        }[self]()
