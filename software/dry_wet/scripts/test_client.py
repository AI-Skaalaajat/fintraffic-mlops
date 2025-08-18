import requests
import os

# --- Configuration ---
# URL of the FastAPI prediction endpoint
# If the serving is running at the kubeflow cluster: 'http://10.106.238.203/dry-wet-model/predict'

API_URL = "http://10.106.238.203/dry-wet-model/predict"

# --- IMPORTANT ---
# test image path.
IMAGE_PATH = "test_image.jpg"

def classify_road_condition(image_path: str):
    """
    Sends an image to the prediction endpoint and prints the result.

    Args:
        image_path (str): The local path to the image file.
    """
    # Check if the image file exists before proceeding
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        print("👉 Please update the IMAGE_PATH variable in this script to a valid file path.")
        return

    # The 'file' key must match the parameter name in the FastAPI endpoint definition
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
            
            print(f"🚀 Sending request to {API_URL} with image: {os.path.basename(image_path)}")
            
            # Send the POST request to the server
            response = requests.post(API_URL, files=files, timeout=30)
            
            # Raise an exception for unsuccessful status codes (4xx or 5xx)
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            print("✅ Prediction successful!")
            print(f"   - Server Response: {result}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to connect to the server at {API_URL}.")
        print(f"   - Please make sure the serving application (app.py) is running.")
        print(f"   - Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure you have the 'requests' library installed:
    # pip install requests
    classify_road_condition(IMAGE_PATH)

