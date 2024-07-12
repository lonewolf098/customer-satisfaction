import logging 
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class model_training(ABC):
    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

class linear_regression_model(model_training):
    def train_model(self, X_train, y_train, **kwargs):
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model 
        except Exception as e:
            raise e