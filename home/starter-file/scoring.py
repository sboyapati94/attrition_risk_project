from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, 'config.json'), 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(CURRENT_DIR, config['output_folder_path']) 
test_data_path = os.path.join(CURRENT_DIR, config['test_data_path']) 
model_path = os.path.join(CURRENT_DIR, config['prod_deployment_path'])

#################Function for model scoring
def score_model():
    """Calculate F1 score of the model on test data"""
    try:
        # Load test data
        test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        
        # Load model
        with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
            model = pickle.load(f)
            
        # Make predictions
        X_test = test_data.drop(['corporation', 'exited'], axis=1)
        y_test = test_data['exited']
        predictions = model.predict(X_test)
        
        # Calculate F1 score
        f1 = metrics.f1_score(y_test, predictions)
        return f1
    except Exception as e:
        print(f"Error in score_model: {str(e)}")
        raise

if __name__ == '__main__':
    score_model()

