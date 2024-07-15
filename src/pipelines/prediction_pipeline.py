import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            ## geting the path of our pickle files
            preprocessor_path = os.path.join('artifact','preprocessor.pkl')
            model_path = os.path.join('artifact','model.pkl')
            
            ## loading pickle files
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            ## scaling the data
            data_scaled = preprocessor.transform(features)
            
            ## making predictions
            pred = model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.error("Error occured during predict pipeline")
            raise CustomException(e,sys)
        
## creating all features as a dataframe which will be used for mapping to front end app  
class CustomData:
    def __init__(self,
                 carat:float,
                 cut:str,
                 color: str,
                 clarity: str,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float):
        
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        
        except Exception as e:
            logging.error("Error in get_data_as_dataframe")
            raise CustomException(e,sys)
        