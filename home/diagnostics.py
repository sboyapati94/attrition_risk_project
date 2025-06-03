import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

##################Load config.json and get environment variables
with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(ROOT_DIR, config['output_folder_path'])
test_data_path = os.path.join(ROOT_DIR, config['test_data_path'])
prod_deployment_path = os.path.join(ROOT_DIR, config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset=None):
    """Get predictions from deployed model"""
    try:
        # If no dataset provided, use test data
        if dataset is None:
            dataset = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
            
        # Load model
        model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Get predictions
        X = dataset.drop(['corporation', 'exited'], axis=1)
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        print(f"Error in model_predictions: {str(e)}")
        raise

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    statistics = []
    for col in numeric_columns:
        statistics.extend([
            df[col].mean(),
            df[col].median(),
            df[col].std()
        ])
    
    return statistics

##################Function to get timings
def execution_time():
    timing = []
    
    # Time for ingestion
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', os.path.join(ROOT_DIR, 'ingestion.py')], capture_output=True)
    timing.append(timeit.default_timer() - starttime)
    
    # Time for training
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', os.path.join(ROOT_DIR, 'training.py')], capture_output=True)
    timing.append(timeit.default_timer() - starttime)
    
    return timing

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated', '--format=json'])
    return json.loads(outdated)

##################Function to get missing data
def missing_data():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    na_percents = df.isna().mean()
    return na_percents.values.tolist()

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data()






