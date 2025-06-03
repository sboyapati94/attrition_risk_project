import sys
import traceback
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from scoring import score_model
    from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    raise

######################Set up variables for use in our script
app = Flask(__name__)

# Get the absolute path of the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Load config
    with open(os.path.join(CURRENT_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    logger.info("Successfully loaded config file")
except Exception as e:
    logger.error(f"Error loading config file: {str(e)}")
    raise

@app.route('/')
def home():
    """Root endpoint to verify API is running"""
    return "Attrition Risk Assessment API is running!"

@app.route("/prediction", methods=['GET'])
def predict():        
    """Return model predictions for test data"""
    try:
        test_data_path = os.path.join(CURRENT_DIR, config['test_data_path'], "testdata.csv")
        logger.info(f"Loading test data from {test_data_path}")
        data = pd.read_csv(test_data_path)
        preds = model_predictions(data)
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/scoring", methods=['GET'])
def scoring():        
    """Return the F1 score of the model on test data"""
    try:
        score = score_model()
        return jsonify({"f1_score": float(score)}), 200
    except Exception as e:
        logger.error(f"Error in scoring endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/summarystats", methods=['GET'])
def stats():        
    """Return summary statistics for the ingested data"""
    try:
        summary = dataframe_summary()
        return jsonify({"summary_statistics": summary}), 200
    except Exception as e:
        logger.error(f"Error in summary stats endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/diagnostics", methods=['GET'])
def diagnostics():        
    """Return various diagnostic information"""
    try:
        timing = execution_time()
        missing = missing_data()
        packages = outdated_packages_list()
        
        return jsonify({
            "execution_time": timing,
            "missing_data": missing,
            "outdated_packages": packages
        }), 200
    except Exception as e:
        logger.error(f"Error in diagnostics endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

def main():
    logger.info("Starting Flask server...")
    try:
        app.run(host='0.0.0.0', port=8000, debug=True)
    except Exception as e:
        logger.error(f"Error starting Flask server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
