import requests
import json
import os

# Load config.json
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

# Specify the URL
URL = "http://127.0.0.1:8000"

# Call prediction endpoint
prediction_response = requests.post(
    f"{URL}/prediction",
    json={"filepath": os.path.join(test_data_path, "testdata.csv")}
).json()

# Call scoring endpoint
scoring_response = requests.get(f"{URL}/scoring").json()

# Call summary statistics endpoint
stats_response = requests.get(f"{URL}/summarystats").json()

# Call diagnostics endpoint
diagnostics_response = requests.get(f"{URL}/diagnostics").json()

# Combine all responses
all_responses = {
    "predictions": prediction_response,
    "scoring": scoring_response,
    "statistics": stats_response,
    "diagnostics": diagnostics_response
}

# Write the responses to a file
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
    json.dump(all_responses, f, indent=4)



