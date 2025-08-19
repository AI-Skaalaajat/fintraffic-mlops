#!/bin/bash
set -e

# This script builds a new version of the serving image and triggers a rolling update.
# Usage: ./update_serving.sh <version>
# Example: ./update_serving.sh v1.0.1

# Check if a version argument is provided
if [ -z "$1" ]; then
    echo "❌ Error: No version tag provided."
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.1"
    exit 1
fi

VERSION=$1
IMAGE_NAME="dry_wet_serving"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"

echo ">>> Writing version to serving/version.txt..."
echo "${VERSION}" > ./serving/version.txt

echo ">>> Configuring docker client to use minikube's docker daemon..."
eval $(minikube -p minikube docker-env)

echo ">>> Building new serving image: '${IMAGE_TAG}'..."
# Build the image from the 'serving' subdirectory
docker build -t ${IMAGE_TAG} -f ./serving/Dockerfile ./serving

echo ">>> Updating deployment to use new image: ${IMAGE_TAG}..."
kubectl set image deployment/dry-wet-model-deployment dry-wet-model=${IMAGE_TAG} -n kubeflow

echo "✅ Done. Deployment is updating."
echo "Monitor status with: kubectl rollout status deployment/dry-wet-model-deployment -n kubeflow"
