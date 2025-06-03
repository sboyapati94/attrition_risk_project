from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list

######################Set up variables for use in our script
app = Flask(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():        
    """Return model predictions for the data file specified in the request"""
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200

    # Get filepath from the request body
    filepath = request.json.get('filepath', None)
    if not filepath:
        return jsonify({"error": "No filepath provided"}), 400
    
    # Read the data
    try:
        data = pd.read_csv(filepath)
        predictions = model_predictions(data)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():        
    """Return the F1 score of the model on test data"""
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200
        
    score = score_model()
    return jsonify({"f1_score": score}), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():        
    """Return summary statistics for the ingested data"""
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200
        
    summary = dataframe_summary()
    return jsonify({"summary_statistics": summary}), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():        
    """Return various diagnostic information"""
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200
        
    timings = execution_time()
    missing = missing_data()
    dependencies = outdated_packages_list()
    
    return jsonify({
        "execution_time": timings,
        "missing_data_percentages": missing,
        "outdated_packages": dependencies
    }), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=False)
