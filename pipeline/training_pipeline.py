from zenml import pipeline
from steps.cleandata import clean_data
from steps.ingestdata import ingest_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model
import os

@pipeline(enable_cache=True)
def data_pipeline(data_path) -> None:
    data = ingest_data(data_path)
    xtrain,xtest,ytrain,ytest = clean_data(data)
    model=train_model(xtrain,xtest,ytrain,ytest)
    mse,r2, rmse = evaluate_model(model, xtest, ytest)