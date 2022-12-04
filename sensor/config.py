import pymongo
import pandas as pd
import json
from dataclasses import dataclass
# Provide the mongodb localhost url to connect python to mongodb.
import os

# This class is to fetch information from the .env file
# To read any information from the environment we need the os module and getenv() of it
# The getenv() takes environment_variable_name as parameter which is stored in the .env file as keys
@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_access_secret_key:str = os.getenv("AWS_SECRET_ACCESS_KEY")


# Creating objects of class EnvironmentVariable
env_var = EnvironmentVariable()

# Creating connection with the mongoDB
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

# Getting the target column
TARGET_COLUMN = "class"