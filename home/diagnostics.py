import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            test_file = os.path.join(test_data_path, "testdata.csv")
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test data not found at {test_file}")
            dataset = pd.read_csv(test_file)
            
        # Load model
        model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found at {model_file}")
            
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        # Get predictions
        X = dataset.drop(['corporation', 'exited'], axis=1)
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        logger.error(f"Error in model_predictions: {str(e)}", exc_info=True)
        raise

##################Function to get summary statistics
def dataframe_summary():
    try:
        data_file = os.path.join(dataset_csv_path, 'finaldata.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data not found at {data_file}")
            
        df = pd.read_csv(data_file)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        statistics = []
        for col in numeric_columns:
            statistics.extend([
                df[col].mean(),
                df[col].median(),
                df[col].std()
            ])
        return statistics
    except Exception as e:
        logger.error(f"Error in dataframe_summary: {str(e)}", exc_info=True)
        raise

##################Function to get timings
def execution_time():
    try:
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
    except Exception as e:
        logger.error(f"Error in execution_time: {str(e)}", exc_info=True)
        raise

##################Function to get missing data
def missing_data():
    try:
        data_file = os.path.join(dataset_csv_path, 'finaldata.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data not found at {data_file}")
            
        df = pd.read_csv(data_file)
        na_percents = df.isna().mean()
        return na_percents.values.tolist()
    except Exception as e:
        logger.error(f"Error in missing_data: {str(e)}", exc_info=True)
        raise

##################Function to check dependencies
def outdated_packages_list():
    try:
        outdated = subprocess.check_output(['pip', 'list', '--outdated', '--format=json'])
        return json.loads(outdated)
    except Exception as e:
        logger.error(f"Error in outdated_packages_list: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    missing_data()






