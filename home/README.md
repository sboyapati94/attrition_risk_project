# Dynamic Risk Assessment System

This project implements a dynamic risk assessment system that can automatically predict which clients are at risk of terminating their contracts. The system includes automated data ingestion, model training, deployment, and monitoring capabilities.

## Project Structure
```
home/
├── ingesteddata/          # Stores ingested and processed data
├── practicemodels/        # Stores model artifacts during development
├── production_deployment/ # Stores deployed model and artifacts
├── sourcedata/           # Source data directory for new datasets
├── testdata/            # Test dataset directory
└── scripts              # Python scripts for the pipeline
```

## Setup Instructions

1. Set up the Python environment:
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate attrition-prediction
```

2. Initialize the pipeline:
```bash
# Run from the home directory
python ingestion.py    # Ingest and process data
python training.py     # Train the model
python scoring.py      # Score the model
python deployment.py   # Deploy the model
python reporting.py    # Generate reports
python app.py         # Start the API server
python apicalls.py    # Test API endpoints
```

3. Set up automated monitoring:
```bash
# Add to crontab
*/10 * * * * cd /Users/user/Downloads/attrition_risk_project/home && /usr/local/bin/python fullprocess.py
```

## API Endpoints

- `/`: Root endpoint to verify API is running
- `/prediction`: Get model predictions
- `/scoring`: Get model F1 score
- `/summarystats`: Get data statistics
- `/diagnostics`: Get model diagnostics

## Files Description

### Python Scripts
- `ingestion.py`: Data ingestion and processing
- `training.py`: Model training
- `scoring.py`: Model scoring
- `deployment.py`: Model deployment
- `diagnostics.py`: Model diagnostics
- `reporting.py`: Generate reports
- `app.py`: Flask API server
- `apicalls.py`: API testing
- `fullprocess.py`: End-to-end process automation

### Output Files
- `finaldata.csv`: Processed data
- `ingestedfiles.txt`: Record of ingested files
- `trainedmodel.pkl`: Trained model
- `latestscore.txt`: Model F1 score
- `confusionmatrix.png`: Model performance visualization
- `apireturns.txt`: API test results

## Automation
The system automatically checks for new data and model drift every 10 minutes using cron jobs.

## Usage
1. Place new data files in the `sourcedata/` directory
2. The automated process will:
   - Detect new data
   - Re-train if needed
   - Check for model drift
   - Update deployment if necessary
   - Generate new reports
