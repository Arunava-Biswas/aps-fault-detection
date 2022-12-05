from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN



class DataTransformation:

    # Doing the initialization
    # Here we need to pass the output of the location where the train and test data is located
    # It is stored in the property named 'data_ingestion_artifact'
    # Another param is the DataTransformationConfig to do the transformation
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


    # Here we are creating the data transformer object and it will return a pipeline
    # We will do it with the class method (so each object of the class will perform same function)
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            # transformations to be applied
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)


    # Now creating the function to transform the data
    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # selecting input features for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Doing label encoding for the target variable to make it to numeric from categorical
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # now doing transformation on target columns of both train and test data
            # it will return array like objects
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)


            # now creating an object of DataTransformation class and apply the transformation function of that class
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            # transforming input features of both train and test data
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            

            # Now doing the sampling to get rid of imbalance data problem
            # Here we want to increase the minority
            smt = SMOTETomek(sampling_strategy="minority")

            # balancing the training data
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            # balancing the test data
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            # target encoder
            # Here we are combining the two arrays returned in after balancing into a single array
            # Basically we are concatinating two arrays for both train and test data
            # In numpy to concatinate we use 'np.c_[arr1, arr2]'
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            # save numpy array using the function created in utils for saving numpy array to file 
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            # saving the transformation object using the function created in utils for saving object to file
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            # saving the label_encoder object using the function created in utils for saving object to file
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)


            # Now preparing the data transformation artifact and saving the above 4 files locations
            # We need to save these locations as we will needed them for model training
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)