import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path  # 'path' should be 'Path'
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)  # Assuming 'artifact_folder' is a valid constant

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()
    
    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)  # Use your actual MongoDB URL
            collection = mongo_client[db_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            
            # Drop the '_id' field if it exists
            if "_id" in df.columns.to_list():
                df = df.drop(columns="_id", axis=1)
            
            # Replace 'na' string with NaN
            df.replace({"na": np.nan}, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store_file_path(self) -> str:  # Return type should be str (path)
        try:
            logging.info("Exporting data from MongoDB")
            raw_file_path = self.data_ingestion_config.artifact_folder

            # Make the directory if it doesn't exist
            os.makedirs(raw_file_path, exist_ok=True)

            # Export data from the MongoDB collection
            sensor_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,  # Assuming these constants are defined
                db_name=MONGO_DATABASE_NAME
            )
            logging.info(f"Saving exported data into feature store file path: {raw_file_path}")

            # Create the file path for saving the CSV
            feature_store_file_path = os.path.join(raw_file_path, 'wafer_fault.csv')

            # Save the DataFrame to CSV
            sensor_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> str:  # Return type should be str (path)
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            # Call the method to export the data and get the file path
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info(f"Got the data from MongoDB and saved to {feature_store_file_path}")

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys) from e
