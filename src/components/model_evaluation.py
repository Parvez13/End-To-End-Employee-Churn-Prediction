from src.predictor import ModelResolver
from src.entity import config_entity, artifact_entity
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
import sys
import os
from src.constant.training_pipeline import TARGET_COLUMN
import numpy as np


class ModelEvaluation:

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact
                 ):
        try:
            logging.info(f"{'>>' * 20}  Model Evaluation {'<<' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("Checking if saved model folder has a model. "
                         "We will compare which model is better: "
                         "the current trained model or the model from the saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                              improved_accuracy=None)
                logging.info(
                    f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            logging.info(
                "Finding the location of the transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info(
                "Loading previously trained objects: transformer, model, and target encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            logging.info("Loading currently trained model objects")
            current_transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(
                file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(
                file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]

            # Handle label encoding for both previous and current target encoders
            y_true = target_encoder.transform(target_df)
            y_pred_previous = np.zeros_like(y_true)  # Initialize with zeros

            # Predict using the previous model
            try:
                input_feature_name = list(transformer.feature_names_in_)
                input_arr = transformer.transform(test_df[input_feature_name])
                y_pred_previous = model.predict(input_arr)
            except ValueError as ve:
                logging.error(
                    f"Error while predicting with the previous model: {ve}")
                # Handle the error gracefully, possibly by using majority class or a default value

            logging.info(
                f"Prediction using the previous model: {y_pred_previous[:5]}")
            previous_model_score = f1_score(
                y_true=y_true, y_pred=y_pred_previous)
            logging.info(
                f"Accuracy using the previous trained model: {previous_model_score}")

            # Predict using the current model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(
                test_df[input_feature_name])
            y_pred_current = current_model.predict(input_arr)
            logging.info(
                f"Prediction using the trained model: {y_pred_current[:5]}")

            current_model_score = f1_score(
                y_true=y_true, y_pred=y_pred_current)
            logging.info(
                f"Accuracy using the current trained model: {current_model_score}")

            if current_model_score <= previous_model_score:
                logging.info(
                    "Current trained model is not better than the previous model")
                raise Exception(
                    "Current trained model is not better than the previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                          improved_accuracy=current_model_score - previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise CustomException(e, sys)
