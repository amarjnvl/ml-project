import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():

            logging.info(f"Training {model_name}")
            if model_name in params and params[model_name]:  # Only do GridSearchCV if we have parameters
                grid_search = GridSearchCV(estimator=model, param_grid=params[model_name], cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(x_train, y_train)  # Fit the model with hyperparameter tuning
                model = grid_search.best_estimator_  # Use the best model from grid search
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                model.fit(x_train, y_train)  # Train the model on the training data without parameter tuning
            logging.info(f"Evaluating {model_name}")

            y_train_pred = model.predict(x_train) # Predict on the training data
            y_test_pred = model.predict(x_test) # Predict on the testing data

            train_model_score = r2_score(y_train, y_train_pred) # Calculate the R2 score for the training data
            test_model_score = r2_score(y_test, y_test_pred) # Calculate the R2 score for the testing data

            report[model_name] = test_model_score

        return report
    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {e}")
        raise CustomException(e, sys)