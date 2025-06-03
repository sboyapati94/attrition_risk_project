from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

###################Load config.json and get path variables
with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(ROOT_DIR, config['output_folder_path'])
model_path = os.path.join(ROOT_DIR, config['output_model_path'])

# Create model directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

#################Function for training the model
def train_model():
    # Load data
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    # Split features and target
    X = data.drop(['corporation', 'exited'], axis=1)
    y = data['exited']
    
    # Train logistic regression model
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                            warm_start=False)
    
    model.fit(X, y)
    
    # Save model
    pickle.dump(model, open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))

if __name__ == '__main__':
    train_model()

