import os
import json
import pandas as pd
import training
import scoring
import deployment
import diagnostics
import reporting
from datetime import datetime

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load config
with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(ROOT_DIR, config['input_folder_path'])
output_folder_path = os.path.join(ROOT_DIR, config['output_folder_path'])
prod_deployment_path = os.path.join(ROOT_DIR, config['prod_deployment_path'])

##################Check and read new data
def check_new_data():
    """Check for new data files"""
    # Read ingestedfiles.txt
    try:
        with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
            ingested_files = set(f.read().splitlines())
    except FileNotFoundError:
        ingested_files = set()

    # Get current files in source directory
    current_files = set([f for f in os.listdir(input_folder_path) if f.endswith('.csv')])

    # Find new files
    new_files = current_files - ingested_files
    return bool(new_files), new_files

##################Checking for model drift
def check_model_drift():
    """Check if model performance has degraded"""
    # Get the current model's score
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        current_score = float(f.read())

    # Get the new score
    new_score = scoring.score_model()

    # Check for significant drift (e.g., 10% degradation)
    return new_score < current_score * 0.9

def main():
    # Check for new data
    has_new_data, new_files = check_new_data()
    if not has_new_data:
        print("No new data found. Exiting...")
        return

    print(f"Found new data files: {new_files}")
    
    # Re-run ingestion with new data
    import ingestion
    ingestion.merge_multiple_dataframe()
    
    # Check for model drift
    if not check_model_drift():
        print("No significant model drift detected. Exiting...")
        return

    print("Model drift detected. Re-training and re-deploying model...")
    
    # Re-train model
    training.train_model()
    
    # Re-deploy model
    deployment.store_model_into_pickle()
    
    # Run diagnostics and reporting
    diagnostics.model_predictions()
    diagnostics.dataframe_summary()
    diagnostics.execution_time()
    diagnostics.missing_data()
    diagnostics.outdated_packages_list()
    reporting.score_model()
    
    print("Full process completed successfully!")

if __name__ == "__main__":
    main()







