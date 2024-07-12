import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class evaluate_model(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(evaluate_model):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
class R2(evaluate_model):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            r2 = r2_score(y_true, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2: {e}")
            raise e 

class RMSE(evaluate_model):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e