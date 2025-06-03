import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset=None):
    """Get predictions from deployed model"""
    # If no dataset provided, use test data
    if dataset is None:
        dataset = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    # Load the deployed model
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Extract features
    X = dataset[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    
    # Make predictions
    predictions = model.predict(X)
    return predictions.tolist()

##################Function to get summary statistics
def dataframe_summary():
    """Calculate summary statistics for each numeric column"""
    # Read the training data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    # Select numeric columns (excluding 'exited' which is the target)
    numeric_columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    
    # Calculate statistics for each column
    summary_stats = []
    for column in numeric_columns:
        stats = [
            df[column].mean(),
            df[column].median(),
            df[column].std()
        ]
        summary_stats.extend(stats)
    
    return summary_stats

##################Function to check for missing data
def missing_data():
    """Calculate percentage of missing values in each column"""
    # Read the training data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    # Calculate NA percentages
    na_percentages = []
    for column in df.columns:
        total = len(df)
        na_count = df[column].isna().sum()
        percentage = (na_count / total) * 100
        na_percentages.append(percentage)
    
    return na_percentages

##################Function to get timings
def execution_time():
    """Calculate timing of training.py and ingestion.py scripts"""
    # Time the ingestion script
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - starttime
    
    # Time the training script
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - starttime
    
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    """Check installed vs latest package versions"""
    # Get list of requirements from requirements.txt
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if '==' in line]
    
    # Run pip list --outdated
    process = subprocess.Popen(['pip', 'list', '--outdated', '--format=json'], 
                             stdout=subprocess.PIPE)
    out, _ = process.communicate()
    outdated = json.loads(out)
    
    # Create list of package info
    package_info = []
    for pkg in outdated:
        package_info.append([
            pkg['name'],
            pkg['version'],
            pkg['latest_version']
        ])
    
    return package_info

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()






