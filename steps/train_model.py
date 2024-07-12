import logging
import pandas as pd
from zenml import step
from src.model_dev import linear_regression_model
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig

@step
def train_model(
    xtrain:pd.DataFrame,
    xtest: pd.DataFrame,
    ytrain: pd.Series,
    ytest: pd.Series,
    config:ModelNameConfig
    )->RegressorMixin:
    try:
        model=None
        if(config.model_name=='linear_regression'):
            model = linear_regression_model()
            train_model=model.train_model(xtrain, ytrain)
            return train_model
    except Exception as e:
        raise e