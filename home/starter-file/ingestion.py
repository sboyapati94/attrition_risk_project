import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

#############Function for data ingestion
def merge_multiple_dataframe():
    # Get all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    
    # Initialize an empty list to store individual dataframes
    dataframes = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Combine all dataframes
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates
        final_df = final_df.drop_duplicates()
        
        # Save the final dataframe
        final_output_path = os.path.join(output_folder_path, "finaldata.csv")
        final_df.to_csv(final_output_path, index=False)
        
        # Save record of ingested files
        record_path = os.path.join(output_folder_path, "ingestedfiles.txt")
        with open(record_path, 'w') as f:
            for file in csv_files:
                f.write(f"{file}\n")
        
        return final_df
    else:
        raise Exception("No CSV files found in the input directory")

if __name__ == '__main__':
    merge_multiple_dataframe()
