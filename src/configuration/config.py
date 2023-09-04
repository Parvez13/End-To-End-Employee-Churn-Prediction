import pymongo
import json
from dataclasses import dataclass
import os


@dataclass
class EnvironmentVariable():
    mongo_db_url: str = os.getenv('MONGO_DB_URL')


env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

# MLFLOw Experimentation
mlflow_tracking_ui: https: // dagshub.com/prvzsohail/End-to-End-Employee-Job-Satisfaction-Project.mlflow
mlflow_USERNAME: prvzsohail
mlflow_PASSWORD: 693d1810cba447c3876af51f98d347c4c14a7972
