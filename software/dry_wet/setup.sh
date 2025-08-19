#!/bin/bash

# ==============================================================================
# Dry-Wet MLOps Setup Script
# ==============================================================================
# This script prepares the local environment for the Kubeflow pipeline by:
# 1. Building the Docker image needed for the training and serving.
# 2. Deploying the model serving component.
# After the setup, you can run the pipeline to train the model and update the model in the serving component.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Preamble ---
echo "🚀 Starting Dry-Wet MLOps Setup Script..."
echo "------------------------------------------------------------------"
sleep 2

# --- 1. Build Docker Image for the Pipeline ---
echo "🏗️  Step 1: Building the training Docker image..."

# For local development, it's crucial to build the image within the context
# of the Kubernetes cluster's Docker daemon. This avoids needing a remote registry.
# The command differs based on your local K8s environment.

if command -v minikube &> /dev/null; then
    echo "Found 'minikube'. Pointing Docker CLI to Minikube's Docker daemon..."
    # This command configures your shell to use the Docker daemon inside Minikube.
    eval $(minikube docker-env)
    echo "Docker environment is now set for Minikube."
fi

# For the initial setup, we build the baseline v1.0.0 version.
IMAGE_NAME="dry_wet_train:v1.0.0"

# Build the image using the Dockerfile in the 'training' directory.
# The build context is './training', so Docker can find all scripts and requirements.
echo "Building image '$IMAGE_NAME' from './training' directory context..."
docker build -t "$IMAGE_NAME" -f ./training/Dockerfile ./training

# If using 'kind', you would now load the image into the cluster nodes:
# echo "If using kind, now run: kind load docker-image $IMAGE_NAME"

echo "✅ Image '$IMAGE_NAME' built successfully."

echo "------------------------------------------------------------------"
sleep 1

# --- 2. Build Serving Image ---
echo "🏗️  Step 2: Building the serving Docker image..."

# For the initial setup, we build the baseline v1.0.0 version.
SERVING_IMAGE_NAME="dry_wet_serving:v1.0.0"

echo "Building image '$SERVING_IMAGE_NAME' from './serving' directory context..."
docker build -t "$SERVING_IMAGE_NAME" -f ./serving/Dockerfile ./serving

echo "✅ Image '$SERVING_IMAGE_NAME' built successfully."
echo "------------------------------------------------------------------"


# --- 3. Deploy Serving Component ---
echo "🚢 Step 3: Deploying the serving component..."

echo "Applying 'serving/serving.yaml' manifest for initial deployment..."
# This applies the manifest as is, for initial setup.
if ! kubectl apply -f ./serving/serving.yaml; then
    echo "❌ Error applying 'serving/serving.yaml'. Please check your kubectl configuration, namespace, and the YAML file."
    exit 1
fi
echo "✅ Serving component deployment initiated."

echo "It may take a few minutes for the service to become ready."
echo "You can check the status with: kubectl get inferenceservice -A"
echo "------------------------------------------------------------------"

echo "🎉 Setup complete! 🎉"
echo "You can now use 'run_pipeline.py' to execute the training pipeline."
