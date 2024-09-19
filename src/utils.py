import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        ## dict to read the model name and evaluation
        report = {}
        
        ## iterating over each model
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Model training
            model.fit(X_train,y_train)
            
            # Model prediction
            pred = model.predict(X_test)
            
            # Model evaluation 
            r2 = r2_score(y_test,pred)
            
            # Saving the evaluation in report
            report[list(models.keys())[i]] = r2
            
        return report
            
            
    except Exception as e:
        logging.error("Error during model_evaluate")
        raise CustomException(e,sys) 
    
def load_object(file_path):
    
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.error("Error during load_object")
        raise CustomException(e,sys)