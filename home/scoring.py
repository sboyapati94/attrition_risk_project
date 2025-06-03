from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#################Load config.json and get path variables
with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    config = json.load(f) 

test_data_path = os.path.join(ROOT_DIR, config['test_data_path'])
model_path = os.path.join(ROOT_DIR, config['output_model_path'])
prod_deployment_path = os.path.join(ROOT_DIR, config['prod_deployment_path'])

#################Function for model scoring
def score_model(model_path=None):
    """Calculate F1 score of the model on test data"""
    try:
        # Load test data
        test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        logger.info(f"Loaded test data from {os.path.join(test_data_path, 'testdata.csv')}")
        
        # Load model from production deployment if not specified
        if model_path is None:
            model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
            
        # Make predictions
        X_test = test_data.drop(['corporation', 'exited'], axis=1)
        y_test = test_data['exited']
        predictions = model.predict(X_test)
        
        # Calculate F1 score
        f1 = metrics.f1_score(y_test, predictions)
        logger.info(f"Calculated F1 score: {f1}")
        
        # Save score to output model path
        score_path = os.path.join(ROOT_DIR, config['output_model_path'], 'latestscore.txt')
        os.makedirs(os.path.dirname(score_path), exist_ok=True)
        with open(score_path, 'w') as f:
            f.write(str(f1))
        logger.info(f"Saved score to {score_path}")
            
        return f1
    except Exception as e:
        logger.error(f"Error in score_model: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    score = score_model()
    print(f"Model F1 score: {score}")

