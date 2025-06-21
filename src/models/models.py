from abc import abstractmethod
import itertools
import time
import numpy as np
from tqdm import tqdm
from typing import List
import attr
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier
from src.models.nn import FeedForwardNet
from src.data_loaders.utils import parquet_cache_for_model
from src.models.evaluation import evaluate_predictions
from src.models.helpers import split_data
from src.data_loaders.datasets import Datasets
from sklearn.dummy import DummyClassifier
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import attr
import pandas as pd


@attr.s
class Model:
    dataset: Datasets = attr.ib()
    kwargs_for_model = attr.ib(factory=dict)
    kwargs_for_splitting_data = attr.ib(factory=dict)

    WHITE_LISTED_KWARGS = ["verbose", "n_jobs"]

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Define the name of the model"""

    @property
    def model_identifier(self) -> str:
        return "_".join(
            [
                self.dataset.value,
                *[
                    f"{key}_{str(val)}"
                    for key, val in self.kwargs_for_splitting_data.items()
                ],
                *[
                    f"{key}_{str(val)}"
                    for key, val in self.kwargs_for_model.items()
                    if key not in self.WHITE_LISTED_KWARGS
                ],
            ]
        )

    @parquet_cache_for_model()
    def run_model(self):
        target, features, df = self.dataset.get_dataset()
        train_df, val_df, test_df = split_data(
            df, target_col=target, **self.kwargs_for_splitting_data
        )

        model_log = f"Training model: {self.model_name} {self.model_identifier}"
        hashes = "".join(len(model_log) * ["#"])
        print("\n" + hashes)
        print(model_log)
        print(hashes + "\n")

        s = time.time()
        model = self.train(train_df[features], train_df[target])
        e = time.time() - s
        print(f"Training took: {e / 60:.2f} minutes")
        print("running predictions...")
        y_pred, y_prob = self.predict(model, val_df[features])
        df = self.evaluate(val_df[target], y_pred, y_prob)
        df["runtime"] = e
        return df

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Define how to train a model"""

    @abstractmethod
    def predict(self, model, X_val: pd.DataFrame):
        """Define how to predict the validation set"""

    @abstractmethod
    def evaluate(self, y_val, y_pred, y_prob):
        print("evaluating...")
        eval = evaluate_predictions(y_val, y_pred, y_prob)
        eval.pop("confusion_matrix")
        eval_df = pd.DataFrame.from_records([eval])
        eval_df["model_name"] = self.model_name
        eval_df["model_id"] = self.model_identifier
        eval_df = eval_df.set_index(["model_name", "model_id"], drop=True)
        print(eval_df)
        return eval_df


@attr.s
class DummyClassifierModel(Model):

    @property
    def model_name(self):
        return "dummy"

    def train(self, X_train, y_train):
        model = DummyClassifier(**self.kwargs_for_model)
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_val):
        return model.predict(X_val), model.predict_proba(X_val)[:, 1]


@attr.s
class LogisticRegressionModel(Model):

    @property
    def model_name(self):
        return "log_reg"

    def train(self, X_train, y_train):
        model = LogisticRegression(**self.kwargs_for_model)
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_val):
        return model.predict(X_val), model.predict_proba(X_val)[:, 1]


@attr.s
class XGBoostModel(Model):

    @property
    def model_name(self):
        return "xgboost"

    def train(self, X_train, y_train):
        model = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **self.kwargs_for_model
        )
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_val):
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        return y_pred, y_prob


@attr.s
class NeuralNetModel(Model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_name(self):
        return "ffnn"

    def train(self, X_train, y_train):
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        y_train_np = y_train.values.astype(np.float32)

        dataset = TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        model = FeedForwardNet(
            X_train_np.shape[1], self.kwargs_for_model.get("hidden_dims")
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(), lr=self.kwargs_for_model.get("lr", 1e-3)
        )
        criterion = nn.BCEWithLogitsLoss()

        num_epochs = 20
        for epoch in range(num_epochs):  # Simple training loop
            model.train()
            loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            print(f"{epoch} / {num_epochs}: {loss:.4f}")
        self._scaler = scaler  # Save for predict
        return model

    def predict(self, model, X_val):
        X_val_np = self._scaler.transform(X_val)
        X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(X_val_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        return preds, probs


class RegisteredModels:

    @classmethod
    def model_registry(cls) -> List[Model]:
        datasets = [
            Datasets.V1,
            Datasets.V2,
            Datasets.V3,
            Datasets.V4,
            Datasets.V5,
        ]

        xgd_boost_params = {
            "n_estimators": [500],
            "max_depth": [8],
            "learning_rate": [0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
        """
        v2_n_estimators_500_max_depth_8_learning_rate_0.1_subsample_0.8_colsample_bytree_0.9
        """

        nn_params = {
            "lr": [0.1, 0.01, 0.001],
            "hidden_dims": [[128, 64], [64, 128, 64, 32]],
        }
        """
        best results seem to be coming from a large hidden layer in the middle and a slow learnings rate
        v2_lr_0.001_hidden_dims_[128, 64, 32]  0.852148   0.867209  0.942980  0.903508  0.900831   483.855843
        v2_lr_0.001_hidden_dims_[128, 64]      0.850207   0.862891  0.946305  0.902675  0.898886  3265.070243
        """
        models = []
        for dataset in datasets:
            if dataset not in [Datasets.V4, Datasets.V5]:
                models.extend(
                    [
                        DummyClassifierModel(
                            dataset=dataset,
                            kwargs_for_model={"strategy": "most_frequent"},
                        ),
                        DummyClassifierModel(
                            dataset=dataset, kwargs_for_model={"strategy": "uniform"}
                        ),
                        LogisticRegressionModel(
                            dataset=dataset,
                            kwargs_for_model={
                                "C": 1.0,
                                "max_iter": 300,
                                "verbose": 1,
                                "n_jobs": -1,
                            },
                        ),
                        LogisticRegressionModel(
                            dataset=dataset,
                            kwargs_for_model={
                                "solver": "saga",
                                "penalty": "l2",  # or 'none' if unregularized
                                "verbose": 1,
                                "max_iter": 300,  # running for anything more than this is prohibitevly expensive
                                "n_jobs": -1,
                            },
                        ),
                        LogisticRegressionModel(
                            dataset=dataset,
                            kwargs_for_model={
                                "solver": "saga",
                                "penalty": "l1",  # or 'none' if unregularized
                                "verbose": 1,
                                "C": 1.0,
                                "max_iter": 300,  # running for anything more than this is prohibitevly expensive
                                "n_jobs": -1,
                            },
                        ),
                    ]
                )
            # Create the cross product
            keys = xgd_boost_params.keys()
            values = xgd_boost_params.values()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            models.extend(
                [
                    XGBoostModel(
                        dataset=dataset,
                        kwargs_for_model={
                            **combo,
                            "n_jobs": -1,
                        },
                    )
                    for combo in combinations
                ]
            )

            keys = nn_params.keys()
            values = nn_params.values()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            models.extend(
                [
                    NeuralNetModel(dataset=dataset, kwargs_for_model=combo)
                    for combo in combinations
                ]
            )

        return models

    @classmethod
    def run_models(cls):
        dfs = []
        models = cls.model_registry()
        for i, model in enumerate(models):
            print(f"############\n{i} / {len(models)}\n#######")
            dfs.append(model.run_model())
        return pd.concat(dfs)
