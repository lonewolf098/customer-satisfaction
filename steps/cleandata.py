import logging
import pandas as pd
from zenml import step
from typing import Tuple
from src.data_cleaning import DataCleaning, DataCleaningStrategy, DataSplitStrategy

@step
def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # Define cleaning strategy
    cleaning_strategy = DataCleaningStrategy()

    # Apply cleaning strategy to data
    cleaned_data = DataCleaning(data, cleaning_strategy).handle_data()

    # Define splitting strategy
    splitting_strategy = DataSplitStrategy()

    # Apply splitting strategy to data
    X_train, X_test, y_train, y_test = splitting_strategy.handle_data(cleaned_data)

    return X_train, X_test, y_train, y_test