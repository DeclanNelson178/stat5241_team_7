from enum import Enum
from typing import List, Tuple
import uuid

import attr
import pandas as pd
from src.data_loaders.rollcall import (
    get_trainig_data_v3,
    get_trainig_data_v4,
    get_training_data_v1,
    get_trainig_data_v2,
    get_training_data_v5,
    get_training_data_v6,
    get_training_data_pass_only,
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

    V6 = "v6"
    """V5 + subjects"""

    PASS_ONLY = "pass_only"

    def get_dataset(self):
        return {
            Datasets.V1: get_training_data_v1,
            Datasets.V2: get_trainig_data_v2,
            Datasets.V3: get_trainig_data_v3,
            Datasets.V4: get_trainig_data_v4,
            Datasets.V5: get_training_data_v5,
            Datasets.V6: get_training_data_v6,
            Datasets.PASS_ONLY: get_training_data_pass_only,
        }[self]()


@attr.s
class DatasetOnTheFly:
    df: pd.DataFrame = attr.ib()
    target: str = attr.ib()

    @property
    def value(self) -> str:
        # random id so nothing gets cached
        return "tmp_" + str(uuid.uuid1())

    def get_dataset(self) -> Tuple[str, List[str], pd.DataFrame]:
        return self.target, [c for c in self.df.columns if c != self.target], self.df
