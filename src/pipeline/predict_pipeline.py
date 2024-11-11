import os
import sys
import pandas as pd
import pickle
from flask import request
from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)


class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            # Save the file
            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Load model and preprocessor
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            # Transform features and make predictions
            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            # Read input data
            input_dataframe = pd.read_csv(input_dataframe_path)

            # Drop unnecessary columns if any
            input_dataframe = input_dataframe.drop(columns="Unnamed: 0", errors='ignore')

            # Make predictions
            prediction_column_name = TARGET_COLUMN
            predictions = self.predict(input_dataframe)

            # Add predictions to dataframe
            input_dataframe[prediction_column_name] = [pred for pred in predictions]

            # Map prediction labels
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            # Save predicted results
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

            logging.info("Predictions completed and saved to: " + self.prediction_pipeline_config.prediction_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        try:
            logging.info("Starting prediction pipeline...")

            # Step 1: Save input file and get the file path
            input_csv_path = self.save_input_files()

            # Step 2: Get predicted dataframe and save it
            self.get_predicted_dataframe(input_csv_path)

            logging.info("Prediction pipeline completed successfully.")
            return self.prediction_pipeline_config.prediction_file_path  # Return the prediction file path

        except Exception as e:
            raise CustomException(e, sys)
