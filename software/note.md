# Delete resources
delete all resources in the namespace:  
`kubectl delete all --all -n kubeflow`

delete the whole namespace:  
`kubectl delete namespace kubeflow`

delete the whole cluster environment

`minikube delete`

# View all pods
`kubectl get pods -A`

# Enable tunnel and view external ip
Enable tunnel:  
`minikube tunnel`

View external ip:  
`kubectl get svc -n istio-system`
