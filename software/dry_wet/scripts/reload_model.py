import requests
import json

# --- Configuration ---
# URL of the FastAPI model reload endpoint

BASE_URL = "http://10.106.238.203/dry-wet-model"
RELOAD_URL = f"{BASE_URL}/model/reload"
INFO_URL = f"{BASE_URL}/model/info"
HEALTH_URL = f"{BASE_URL}/"

def check_health():
    """Check if the service is running and model status."""
    try:
        print(f"🔍 Checking service health at {HEALTH_URL}")
        response = requests.get(HEALTH_URL, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print("📊 Service Status:")
        print(f"   - Status: {result.get('status', 'unknown')}")
        print(f"   - Model Loaded: {result.get('model_loaded', 'unknown')}")
        print(f"   - Model Version: {result.get('model_version', 'unknown')}")
        print(f"   - Model Stage: {result.get('model_stage', 'unknown')}")
        return result.get('model_loaded', False)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to connect to the service at {HEALTH_URL}")
        print(f"   - Details: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during health check: {e}")
        return False

def get_model_info():
    """Get detailed model information."""
    try:
        print(f"📋 Getting model info from {INFO_URL}")
        response = requests.get(INFO_URL, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print("📊 Current Model Info:")
        for key, value in result.items():
            print(f"   - {key}: {value}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to get model info from {INFO_URL}")
        print(f"   - Details: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error getting model info: {e}")
        return None

def reload_model():
    """
    Send a request to reload the model from MLflow Model Registry.
    """
    try:
        print(f"🔄 Sending reload request to {RELOAD_URL}")
        
        # Send the POST request to reload the model
        response = requests.post(RELOAD_URL, timeout=60)  # Longer timeout for model loading
        
        # Raise an exception for unsuccessful status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        print("✅ Model reload request successful!")
        print(f"   - Status: {result.get('status', 'unknown')}")
        print(f"   - Message: {result.get('message', 'No message provided')}")
        print(f"   - Model Version: {result.get('model_version', 'unknown')}")
        
        return True
        
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out. Model loading may take longer than expected.")
        print(f"   - Try checking the service status after a few minutes.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Failed to connect to the server at {RELOAD_URL}")
        print(f"   - Please make sure the serving application is running.")
        print(f"   - Details: {e}")
        
        # Try to get more details from the response if available
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"   - Server Error Details: {json.dumps(error_details, indent=2)}")
            except:
                print(f"   - Server Response Text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")
        return False

def main():
    """Main function to orchestrate the model reload process."""
    print("=" * 60)
    print("🤖 MLflow Model Reload Client")
    print("=" * 60)
    
    # Step 1: Check service health
    print("\n📡 Step 1: Checking service health...")
    is_healthy = check_health()
    
    if not is_healthy:
        print("\n⚠️  Service appears to be down or model is not loaded.")
        print("   Proceeding with reload attempt anyway...")
    
    # Step 2: Get current model info (optional, before reload)
    print("\n📋 Step 2: Getting current model information...")
    old_info = get_model_info()
    
    # Step 3: Reload the model
    print("\n🔄 Step 3: Reloading model...")
    reload_success = reload_model()
    
    if reload_success:
        print("\n📋 Step 4: Getting updated model information...")
        # Give the server a moment to complete the reload
        import time
        time.sleep(2)
        new_info = get_model_info()
        
        # Compare versions if both are available
        if old_info and new_info:
            old_version = old_info.get('model_version')
            new_version = new_info.get('model_version')
            if old_version and new_version:
                if old_version != new_version:
                    print(f"🆕 Model version changed from {old_version} to {new_version}")
                else:
                    print(f"📌 Model version remains the same: {new_version}")
    
    print("\n" + "=" * 60)
    print("✨ Model reload process completed!")
    print("=" * 60)
    
    return reload_success

if __name__ == "__main__":
    # Ensure you have the 'requests' library installed:
    # pip install requests
    success = main()
    
    # Exit with appropriate code
    exit(0 if success else 1)
