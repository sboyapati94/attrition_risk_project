import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Get the absolute path of the directory containing the script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#############Load config.json and get input and output paths
with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(ROOT_DIR, config['input_folder_path'])
output_folder_path = os.path.join(ROOT_DIR, config['output_folder_path'])

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

#############Function for data ingestion
def merge_multiple_dataframe():
    # Get all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    
    # Record ingested files
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        for file in csv_files:
            f.write(f"{file}\n")
    
    # Read and combine all CSV files
    df_list = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(input_folder_path, file))
        df_list.append(df)
    
    # Combine all dataframes
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # Drop duplicates
    final_df = final_df.drop_duplicates()
    
    # Save final dataframe
    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    
    return final_df

if __name__ == '__main__':
    df = merge_multiple_dataframe()
