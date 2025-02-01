from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any] = None):
        self.model_params = model_params or {}
        self.model = None
        self.best_params = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return rmse, r2
