import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE,R2,RMSE
from typing_extensions import Annotated,Tuple

@step
def evaluate_model(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.DataFrame)-> Tuple[Annotated[float,"mse"],Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        predictions = model.predict(x_test)
        mse = MSE().calculate_scores(y_test, predictions)
        r2 = R2().calculate_scores(y_test, predictions)
        rmse = RMSE().calculate_scores(y_test, predictions)
        logging.info(f"Model evaluation metrics: MSE={mse}, R2={r2}, RMSE={rmse}")
        return mse, r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
