# Basic Import
import numpy as np
import pandas as pd
import os,sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trainer_obj_file = os.path.join("artifact",'model.pkl')
    
class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting dependent and independent variables")
            
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            print(f"Train Set {X_train.shape, y_train.shape} \n")
            print(f"Test Set {X_test.shape, y_test.shape} \n")
            
            ## Define all the models
            models = {
                'Linear_Regression' : LinearRegression(),
                'Lasso' : Lasso(),
                'Ridge' : Ridge(),
                'Elastic_Net' : ElasticNet(),
                'Decision_Tree' : DecisionTreeRegressor(),
                'Random_Forest' : RandomForestRegressor(),
                'Gradient_Boosting': GradientBoostingRegressor(),
                'Ada_Boost' : AdaBoostRegressor(),
                'XGboost' : XGBRegressor()
            }
            
            ## model evaluation
            model_report= dict(evaluate_model(X_train,X_test, y_train, y_test, models))
            print(model_report)
            print('\n========================================================================')
            logging.info(f'Model Report : {model_report}')
            
            
            
            ## finding best model
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print('\n===============================================================================')
            logging.info(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')
            
            save_object(
                file_path = self.model_trainer_config.trainer_obj_file,
                obj=best_model
            )
            
            
        except Exception as e:
            logging.error("Exception occured during initate_model_training")
            raise CustomException(e,sys)
