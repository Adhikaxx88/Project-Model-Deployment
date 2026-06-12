from abc import ABC, abstractmethod

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

from config import (
    LGBM_PARAM_GRID,
    LGBM_PARAMS,
    MLP_PARAM_GRID,
    MLP_PARAMS,
    RF_PARAM_GRID,
    RF_PARAMS,
    SEARCH_PARAMS,
    XGB_PARAM_GRID,
    XGB_PARAMS,
)


class BaseTrainer(ABC):
    """Common interface for a model trained on already-preprocessed features."""

    _name: str

    def __init__(self) -> None:
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        """Return an unfitted sklearn-compatible estimator."""

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseTrainer":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def name(self) -> str:
        return self._name

    @property
    def final_estimator(self):
        """The fitted estimator to embed in the deployment pipeline."""
        return self.model


class RandomForestTrainer(BaseTrainer):
    _name = "Random Forest"

    def build_model(self):
        return RandomForestClassifier(**RF_PARAMS)


class XGBoostTrainer(BaseTrainer):
    _name = "XGBoost"

    def build_model(self):
        return xgb.XGBClassifier(**XGB_PARAMS)


class LightGBMTrainer(BaseTrainer):
    _name = "LightGBM"

    def build_model(self):
        return lgb.LGBMClassifier(**LGBM_PARAMS)


class MLPTrainer(BaseTrainer):
    _name = "MLP"

    def build_model(self):
        return MLPClassifier(**MLP_PARAMS)


class TunedTrainer(BaseTrainer):
    """Wraps a base estimator with RandomizedSearchCV hyperparameter tuning."""

    def __init__(self, name: str, base_estimator, param_grid: dict) -> None:
        self._name = name
        self._base_estimator = base_estimator
        self._param_grid = param_grid
        super().__init__()

    def build_model(self):
        return RandomizedSearchCV(
            self._base_estimator,
            param_distributions=self._param_grid,
            **SEARCH_PARAMS,
        )

    @property
    def best_params_(self) -> dict:
        return self.model.best_params_

    @property
    def final_estimator(self):
        return self.model.best_estimator_


def get_all_trainers() -> list[BaseTrainer]:
    return [RandomForestTrainer(), XGBoostTrainer(), LightGBMTrainer(), MLPTrainer()]


def get_tuned_trainers() -> list[BaseTrainer]:
    return [
        TunedTrainer(
            "Random Forest (Tuned)",
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            RF_PARAM_GRID,
        ),
        TunedTrainer(
            "XGBoost (Tuned)",
            xgb.XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1),
            XGB_PARAM_GRID,
        ),
        TunedTrainer(
            "LightGBM (Tuned)",
            lgb.LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
            LGBM_PARAM_GRID,
        ),
        TunedTrainer(
            "MLP (Tuned)",
            MLPClassifier(max_iter=300, random_state=42),
            MLP_PARAM_GRID,
        ),
    ]
