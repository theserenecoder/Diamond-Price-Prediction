from sklearn.impute import SimpleImputer  ## Handling missing values
from sklearn.preprocessing import StandardScaler ## handling feature scaling
from sklearn.preprocessing import OrdinalEncoder ## handling orinal encoding

## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocssor_obj_file = os.path.join("artifact",'preprocessor.pkl')


## Data Ingestion config class

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation initiated")
            
            # Define which columns should be ordinal-encoded and which are numerical
            categorical_columns = ['cut', 'color', 'clarity'] 
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            # Define custom ranking for each ordial variable
            cut_category = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            clarity_category = ['I1' , 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            color_category = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            
            # numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaling',StandardScaler())
                ]
            )
            
            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('encoder',OrdinalEncoder(categories=[cut_category,color_category,clarity_category])),
                    ('scaling', StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return preprocessor

            logging.info("Pipeline completed")
            
        except Exception as e:
            
            logging.error("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_data_path,test_data_path):
        
        try:
            # reading train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info('Reading train and test data completed')
            logging.info(f'Train Dataset Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataset Head : \n {test_df.head().to_string()}')
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformation_obj()
            
            target_column = 'price'
            drop_columns = [target_column,'id']
            
            # features into independent and dependent features
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_features_test_df = test_df.drop(columns = drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]
            
            ## apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            logging.info("Applying preprocessing object on train and test datasets")
            
            ## concatinating the test and train data as array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            ## calling save_object from utils to save the preprocessing pickel file
            save_object(
                file_path=self.data_transformation_config.preprocssor_obj_file,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessor pickel is created")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocssor_obj_file
            )
        
        except Exception as e:
            logging.error("Exception occured in the initiate_Datatransformation")
            raise CustomException(e,sys)