import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def get_data(self):
        logging.info("Ingesting data from {}".format(self.data_path))
        try:
            data = pd.read_csv(self.data_path)
            return data
        except FileNotFoundError:
            logging.error("Failed to find data file at {}".format(self.data_path))
            raise
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file."""
    ingestor = IngestData(data_path)
    return ingestor.get_data()