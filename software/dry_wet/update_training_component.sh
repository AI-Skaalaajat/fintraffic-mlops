#!/bin/bash
set -e

# This script builds a new version of the training image.
# You can then use this new image tag when running the Kubeflow pipeline.
# Usage: ./update_training_image.sh <version>
# Example: ./update_training_image.sh v1.0.1

# Check if a version argument is provided
if [ -z "$1" ]; then
    echo "❌ Error: No version tag provided."
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.1"
    exit 1
fi

VERSION=$1
IMAGE_NAME="dry_wet_train"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"

echo ">>> Writing version to training/version.txt..."
echo "${VERSION}" > ./training/version.txt

echo ">>> Configuring docker client to use minikube's docker daemon..."
eval $(minikube -p minikube docker-env)

echo ">>> Building new training image: '${IMAGE_TAG}'..."
# Build the image from the 'training' subdirectory
docker build -t ${IMAGE_TAG} -f ./training/Dockerfile ./training

echo "✅ Done. New training image '${IMAGE_TAG}' is built and available in minikube."
echo "You can now use this tag with the '--training-image' argument when running your pipeline."
