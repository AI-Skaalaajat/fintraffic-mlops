#!/bin/bash
# This script cleans up Kubernetes pods that meet the following criteria:
# 1. The pod name starts with 'dry-wet-model-training'.
# 2. The pod's status is 'Completed' (phase 'Succeeded') or 'Error' (phase 'Failed').

set -e

echo "Searching for pods to clean up..."

# Get all pods in 'Succeeded' or 'Failed' phases across all namespaces.
# The output format is "<namespace> <pod-name>".
# Then, filter for pods with names starting with 'dry-wet-model-training'.
# The '|| true' part prevents the script from exiting if grep finds no matches.
PODS_TO_DELETE=$(kubectl get pods --all-namespaces -o go-template='{{range .items}}{{if or (eq .status.phase "Succeeded") (eq .status.phase "Failed")}}{{.metadata.namespace}}{{" "}}{{.metadata.name}}{{"\n"}}{{end}}{{end}}' | grep 'dry-wet-model-training' || true)

if [ -z "$PODS_TO_DELETE" ]; then
  echo "No 'Completed' or 'Error' pods found with prefix 'dry-wet-model-training'."
  exit 0
fi

echo "Found the following pods to delete:"
echo "$PODS_TO_DELETE"
echo

# Delete the pods. The xargs command processes pairs of <namespace> <pod-name>.
# For each pair, it calls 'kubectl delete pod <pod-name> -n <namespace>'.
# The -r flag for xargs prevents it from running if there's no input.
echo "$PODS_TO_DELETE" | xargs -n 2 -r bash -c 'echo "Deleting pod $1 in namespace $0..." && kubectl delete pod "$1" -n "$0"'

echo "Cleanup finished."
