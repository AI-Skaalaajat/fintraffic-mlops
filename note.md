# Delete resources

### Delete all resources in the namespace
`kubectl delete all --all -n kubeflow`

### Delete the whole namespace
`kubectl delete namespace kubeflow`

### Delete the whole cluster environment
`minikube delete`

# Pods

### View all pods
`kubectl get pods -A`

### Check the logs of a pod
`kubectl logs <pod-name> -n <namespace>`

### Check the description of a pod
`kubectl describe pod <pod-name> -n <namespace>`

# Enable tunnel and view external ip
Enable tunnel:  
`minikube tunnel`

View external ip:  
`kubectl get svc -n istio-system`

