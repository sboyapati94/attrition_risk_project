import requests
import json
import os
import time

# Get the absolute path of the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def call_api_endpoint(url, endpoint, max_retries=3):
    """Helper function to call API endpoint with retries"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i+1}/{max_retries} failed for {endpoint}: {str(e)}")
            if i < max_retries - 1:  # don't sleep on the last attempt
                time.sleep(2)
    return None

def check_api_available(url, timeout=30):
    """Check if API is available, wait up to timeout seconds"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("API is available!")
                return True
        except requests.exceptions.RequestException:
            pass
        print("Waiting for API to become available...")
        time.sleep(2)
    return False

def save_api_responses(responses, config, suffix=""):
    """Save API responses to file"""
    output_model_path = os.path.join(CURRENT_DIR, config['output_model_path'])
    os.makedirs(output_model_path, exist_ok=True)
    
    output_file = os.path.join(output_model_path, f'apireturns{suffix}.txt')
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"API responses saved to {output_file}")

if __name__ == "__main__":
    # Load config
    with open(os.path.join(CURRENT_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # API URL
    URL = "http://127.0.0.1:8000"
    
    # Check if API is available
    if not check_api_available(URL):
        print("ERROR: API is not available")
        exit(1)

    # List of endpoints to test
    endpoints = [
        "/prediction",
        "/scoring",
        "/summarystats",
        "/diagnostics"
    ]

    # Call each endpoint and store responses
    responses = {}
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        response = call_api_endpoint(URL, endpoint)
        if response:
            print(f"Success: {json.dumps(response, indent=2)}")
            responses[endpoint] = response
        else:
            print(f"Failed to get response from {endpoint}")
            responses[endpoint] = {"error": "Failed to get response"}

    # Save responses to both files
    save_api_responses(responses, config)  # Original file
    save_api_responses(responses, config, "2")  # Second file
    print("\nAPI testing completed!")



