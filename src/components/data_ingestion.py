import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact', 'test.csv')
    raw_data_path = os.path.join('artifact','raw.csv')

## create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_Data_ingestion(self):
        #logging.info("Data Ingestion method start")
        
        try:
            ## reading data into dataframe from file
            df=pd.read_csv(os.path.join('notebooks\data','gemstone.csv'))
            logging.info('Dataset read as pandas dataframe')
            
            ## creating artifact folder if not created
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            ## saving the raw data in raw data file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw Datafile saved in artifact folder")
            
            ## train test dataset split
            train_set, test_set = train_test_split(df,test_size=.30, random_state=42)
            logging.info('Train test split completed')
            
            ## saving the train set data in file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            logging.info("Training data saved in train.csv")
            
            ## saving the test set data in file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Test data saved in test.csv ")
            
            logging.info('Ingestion of data is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error("Error occured in Data Ingestion config step")
        
