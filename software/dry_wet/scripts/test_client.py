import requests
import os
import argparse

# --- Configuration ---
# URL of the FastAPI prediction endpoint
# If the serving is running at the kubeflow cluster: 'http://<external-ip>/dry-wet-model/predict'


def check_service_status(base_url: str):
    """
    Performs a health check and retrieves model information from the server.
    Args:
        base_url (str): The base URL of the prediction service.
    Returns:
        bool: True if the service is healthy and info is retrieved, False otherwise.
    """
    try:
        # --- Health Check ---
        health_url = base_url.rstrip('/') + '/'
        print(f"🩺 Checking service health at {health_url}...")
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        health_data = response.json()
        print("✅ Service is OK.")
        print(f"   - Status: {health_data.get('status')}")
        print(f"   - Model Loaded: {health_data.get('model_loaded')}")

        if not health_data.get('model_loaded'):
            print("   - ⚠️ Warning: Model is not loaded on the server.")
            return False

        # --- Model Info ---
        info_url = base_url.rstrip('/') + '/model/info'
        print(f"\nℹ️  Fetching model information from {info_url}...")
        response = requests.get(info_url, timeout=10)
        response.raise_for_status()
        info_data = response.json()
        print("✅ Model information received:")
        print(f"   - Model Name: {info_data.get('model_name')}")
        print(f"   - Model Version: {info_data.get('model_version')}")
        print(f"   - Model Stage: {info_data.get('model_stage')}")
        print(f"   - Description: {info_data.get('model_description')}")
        print("-" * 30)
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to the service at {base_url}.")
        print("   - Please check if the serving application is running and the IP is correct.")
        print(f"   - Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def classify_road_condition(image_path: str, api_url: str):
    """
    Sends an image to the prediction endpoint and prints the result.

    Args:
        image_path (str): The local path to the image file.
        api_url (str): The URL of the prediction endpoint.
    """
    # Check if the image file exists before proceeding
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        print("👉 Please provide a valid file path for the image.")
        return

    # The 'file' key must match the parameter name in the FastAPI endpoint definition
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
            
            print(f"🚀 Sending request to {api_url} with image: {os.path.basename(image_path)}")
            
            # Send the POST request to the server
            response = requests.post(api_url, files=files, timeout=30)
            
            # Raise an exception for unsuccessful status codes (4xx or 5xx)
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            print("✅ Prediction successful!")
            print(f"   - Server Response: {result}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to connect to the server at {api_url}.")
        print(f"   - Please make sure the serving application (app.py) is running.")
        print(f"   - Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure you have the 'requests' library installed:
    # pip install requests
    parser = argparse.ArgumentParser(description="Classify road condition from an image.")
    parser.add_argument(
        "--api-ip",
        type=str,
        default="localhost",
        help="IP address of the FastAPI prediction endpoint."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="test_image.jpg",
        help="The local path to the image file."
    )
    args = parser.parse_args()
    
    # Construct base URL for the service
    base_url = f"http://{args.api_ip}/dry-wet-model"

    # 1. Check service status and get model info
    print("--- Step 1: Checking Service Status ---")
    if check_service_status(base_url):
        # 2. If status is OK, proceed with prediction
        print("\n--- Step 2: Classifying Image ---")
        predict_url = f"{base_url}/predict"
        classify_road_condition(image_path=args.image_path, api_url=predict_url)

