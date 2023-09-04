from src.entity import artifact_entity,config_entity
from src.exception import CustomException
from src.logger import logging
from typing import Optional
import os,sys 
from sklearn.ensemble import RandomForestClassifier
from src.utils import main_utils
from sklearn.metrics import f1_score
import mlflow 
import mlflow.sklearn


class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    # def fine_tune(self):
    #     try:
    #         #Wite code for Grid Search CV
    #         pass
            

        # except Exception as e:
        #     raise CustomException(e, sys)

    def train_model(self,x,y):
        try:
            rf_clf = RandomForestClassifier(min_samples_leaf=4, min_samples_split=10,
                       n_estimators=300, random_state=42)
            rf_clf.fit(x,y)
            return rf_clf
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            mlflow.start_run()
            logging.info(f"Loading train and test array.")
            train_arr = main_utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = main_utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score  =f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score  =f1_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            main_utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
            # Log metrics with MLflow
            mlflow.log_metric("f1_train_score", f1_train_score)
            mlflow.log_metric("f1_test_score", f1_test_score)

            # Save the trained model with MLflow
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            mlflow.end_run()
            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)