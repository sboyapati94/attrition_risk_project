from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

# Create production deployment directory if it doesn't exist
os.makedirs(prod_deployment_path, exist_ok=True)

####################function for deployment
def store_model_into_pickle():
    # Copy the trained model
    shutil.copy2(
        os.path.join(model_path, 'trainedmodel.pkl'),
        os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    )
    
    # Copy the latestscore
    shutil.copy2(
        os.path.join(model_path, 'latestscore.txt'),
        os.path.join(prod_deployment_path, 'latestscore.txt')
    )
    
    # Copy the ingested files record
    shutil.copy2(
        os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
        os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    )

if __name__ == '__main__':
    store_model_into_pickle()




