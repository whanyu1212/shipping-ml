from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from rental_prediction.config import ModelConfig
from rental_prediction.modelling.base_model import BaseModel


class RfModel(BaseModel):
    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__(model_params)
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = RandomForestRegressor(**self.model_params)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
    ) -> Dict[str, Any]:

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }

            self.model_params = params
            self.train(X_train, y_train)
            rmse, _ = self.evaluate(X_val, y_val)
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.model_params = self.best_params
        self.train(X_train, y_train)
        return self.best_params


if __name__ == "__main__":
    df = pd.read_csv("/workspaces/ml-in-prod-example/data/rent_apartments_processed.csv")
    X = df[
        [
            "area",
            "constraction_year",
            "bedrooms",
            "garden",
            "balcony_yes",
            "parking_yes",
            "furnished_yes",
            "garage_yes",
            "storage_yes",
        ]
    ]
    y = df["rent"]
    model = RfModel()
    config = ModelConfig()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.val_size
    )
    model.optimize(X_train, y_train, X_val, y_val, n_trials=100)
    print(model.best_params)
