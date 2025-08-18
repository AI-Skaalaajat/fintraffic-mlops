import os
import io
import yaml
import mlflow
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# --- MLflow Configuration ---
# Configure MLflow to connect to the same backend as training
def configure_mlflow():
    """Configure MLflow tracking URI and S3 endpoint for model registry."""
    # Set MLflow tracking URI - should point to the same MLflow server used in training
    # In Kubeflow environment, this is typically the MLflow service
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server-service.kubeflow.svc.cluster.local:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Configure S3 endpoint for MLflow artifact storage (MinIO)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.kubeflow.svc.cluster.local:9000")
    
    print(f"MLflow configured with tracking URI: {tracking_uri}")
    print(f"MLflow S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

# Initialize MLflow configuration
configure_mlflow()

# --- Configuration ---
def load_config(config_path="config.yaml"):
    """Load configuration for transformations."""
    # This config should ideally be packaged with the model in MLflow,
    # but for now, we'll load it from a local file.
    # We'll need to add this file to our Docker image.
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        # Fallback to default values if config is not found
        print(f"Warning: {config_path} not found. Using default transformation values.")
        return {
            'training': {'img_size': 224},
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }

config = load_config()
img_size = config['training']['img_size']
normalization_mean = config['normalization']['mean']
normalization_std = config['normalization']['std']

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Dry/Wet Road Classifier",
    description="Accepts an image and returns 'dry' or 'wet' classification.",
    version="1.0.0"
)

# --- Version Loading ---
def load_image_version(version_file="version.txt"):
    """Load the image version from a file."""
    try:
        with open(version_file, 'r') as f:
            version = f.read().strip()
        return version
    except FileNotFoundError:
        return "unknown"

image_version = load_image_version()

# --- Model Loading ---
# Load the model from MLflow Model Registry at startup.
# This makes inference faster as the model is already in memory.
model_name = "dry_wet_model"
model_stage = "Production"
model = None
model_version = None
model_info = None

def load_model():
    """Load model from MLflow Model Registry."""
    global model, model_version, model_info
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        print(f"Loading model from URI: {model_uri}")
        
        # Get model info before loading
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get the latest Production model version
        latest_versions = client.get_latest_versions(model_name, stages=[model_stage])
        if not latest_versions:
            raise Exception(f"No model found in {model_stage} stage for {model_name}")
        
        model_info = latest_versions[0]
        model_version = model_info.version
        
        model = mlflow.pytorch.load_model(model_uri)
        model.eval() 
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✅ Model loaded successfully - Version: {model_version}")
        return True
    except Exception as e:
        print(f"❌ Fatal: Could not load model. Error: {e}")
        model = None
        model_version = None
        model_info = None
        return False

# Load model at startup
load_model()

# --- Image Transformations ---
# Define the same transformations as used in validation.
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_mean, std=normalization_std),
])

# Define class names - this should match your training labels.
# Ideally, this should also be an artifact logged with the model.
class_names = ['dry', 'wet']

# --- API Endpoints ---
@app.get("/", summary="Health Check", description="Check if the service is running.")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_version": model_version,
        "model_stage": model_stage,
        "image_version": image_version
    }

@app.get("/model/info", summary="Model Information", description="Get information about the currently loaded model.")
def get_model_info():
    """Get detailed information about the currently loaded model."""
    if not model:
        return JSONResponse(status_code=404, content={"error": "No model loaded"})
    
    return {
        "model_name": model_name,
        "model_version": model_version,
        "model_stage": model_stage,
        "model_description": getattr(model_info, 'description', 'No description available'),
        "creation_timestamp": getattr(model_info, 'creation_timestamp', None),
        "last_updated_timestamp": getattr(model_info, 'last_updated_timestamp', None)
    }

@app.post("/model/reload", summary="Reload Model", description="Reload the model from MLflow Model Registry.")
def reload_model():
    """Reload the model from MLflow Model Registry."""
    success = load_model()
    if success:
        return {
            "message": "Model reloaded successfully",
            "model_version": model_version,
            "status": "success"
        }
    else:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Failed to reload model",
                "status": "error"
            }
        )

@app.post("/predict", summary="Classify Road Condition", description="Upload an image to classify if the road is 'dry' or 'wet'.")
async def predict(file: UploadFile = File(...)):
    """
    Takes an image file, preprocesses it, and returns the predicted label.
    """
    # Use a try-except block to catch any exceptions during prediction
    # and return a detailed error message.
    try:
        if not model:
            return JSONResponse(
                status_code=500,
                content={"error": "Model is not loaded, cannot perform prediction."}
            )

        # Read image data
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply transformations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = class_names[predicted_idx.item()]

        return {"prediction": predicted_label}

    except Exception as e:
        # If any error occurs, log it and return a 500 error with details.
        # This provides much better debugging information than a generic error.
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred during prediction.",
                "error_type": str(type(e).__name__),
                "error_details": str(e)
            }
        )

# --- Main execution ---
if __name__ == "__main__":
    # The production environment will use a Gunicorn or Uvicorn server.
    uvicorn.run(app, host="0.0.0.0", port=5001) 