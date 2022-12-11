import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from datetime import datetime

# defining the variables
FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

# to create an artifact folder and a timestamp folder in the current directory
class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise SensorException(e,sys)     


# these are the input classes
class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="aps"
            self.collection_name="sensor"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise SensorException(e,sys)     

    # to convert the dataset as dictionary and return it as dictionary format
    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise SensorException(e,sys)     


class DataValidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        # Creating validation reports file which will contain all the detailed report about the dataset
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
        # We can create reports in both .yaml and .json format
        self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
        # Setting the threshold for missing values in column
        self.missing_threshold:float = 0.2
        # Giving the base file's location to do the validation
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")


class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        # Creating the transformation directory to store the 3 things
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
        # Creating the location of the transformation object
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        # Creating location of transformed train data
        # We are saving the transformed file in .npz file format
        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
        # Creating location of transformed test data
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))
        # Creating location for the target encoder
        self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)


class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        # Setting the expected score for F1 score 
        self.expected_score = 0.7
        # Setting the overfitting threshold (10%)
        self.overfitting_threshold = 0.1


class ModelEvaluationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.01


# here '...' is similar to the 'pass'
class ModelPusherConfig:

     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir , "model_pusher")
        self.saved_model_dir = os.path.join("saved_models")
        # This will create a saved_models folder outside the artifact folder
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)