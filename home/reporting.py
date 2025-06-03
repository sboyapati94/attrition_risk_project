import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

###############Load config.json and get path variables
with open(os.path.join(ROOT_DIR, 'config.json'),'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(ROOT_DIR, config['output_folder_path'])
test_data_path = os.path.join(ROOT_DIR, config['test_data_path'])
output_model_path = os.path.join(ROOT_DIR, config['output_model_path'])




##############Function for reporting
def score_model():
    """Calculate and plot confusion matrix"""
    # Read test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    # Get predictions
    y_pred = model_predictions()
    y_true = test_data['exited']
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # Create the confusion matrix plot
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))
    plt.close()

if __name__ == '__main__':
    score_model()
