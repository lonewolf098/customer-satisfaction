from pipeline.training_pipeline import data_pipeline
import os
if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data", "data.csv")
    pipeline_instance = data_pipeline(data_path)
