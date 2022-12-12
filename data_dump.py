import pymongo
import pandas as pd
import json
from sensor.config import mongo_client

print(f"Loading environment variables from .env file")
load_dotenv()

# Provide the mongodb localhost url to connect python to mongodb.
# When we import the mongo_client then we don't need this hardcoded link anymore
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

# The data path
DATA_FILE_PATH = "/config/workspace/aps_failure_training_set1.csv"

# Creating database and collection names
DATABASE_NAME="aps"
COLLECTION_NAME="sensor"


if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)