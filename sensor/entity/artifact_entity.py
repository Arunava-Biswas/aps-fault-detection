from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    # This variable will hold the location for the report.yaml file
    report_file_path:str


@dataclass
class DataTransformationArtifact:
    # outputs from the data transformation class from config_entity file
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_encoder_path:str


@dataclass
class ModelTrainerArtifact:
    # Here we will have the model path and f1 score for both train and test data
    model_path:str 
    f1_train_score:float 
    f1_test_score:float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float


class ModelPusherArtifact:...
