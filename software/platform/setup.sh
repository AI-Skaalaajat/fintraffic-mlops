set -e
# Yq is used to read the values from the configuration file. 
# This needs to be installed for the config to be usable

if ! command -v yq >/dev/null 2>&1
then
    echo "🔍 yq not found, installing temporarily..."
    wget -qO yq "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64"
    chmod +x yq
fi

#
# The values from the configuration file
#
CONFIG_FILE="config.yaml"


PIPELINES_VERSION=$(./yq  '.kubeflow.pipelines_version' $CONFIG_FILE)
BUCKETS=$(./yq '.minio.buckets[]' $CONFIG_FILE)
GRAFANA_URL=$(./yq '.monitoring.grafana_url' $CONFIG_FILE)
GRAFANA_PASSWORD=$(./yq '.monitoring.grafana_password' $CONFIG_FILE)
PROMETHEUS_URL=$(./yq '.monitoring.prometheus_url' $CONFIG_FILE)
MINIO_USER=$(./yq '.minio.user' $CONFIG_FILE)
MINIO_PASSWORD=$(./yq '.minio.password' $CONFIG_FILE)

#
# This will install Istio, which is used as a service mesh but also as a gateway 
# for routing virtual services
# This allows us to create gateways to different services using one load balancer IP, example:
# <external-ip>/kubeflow - Will redirect you to kubeflow
#
echo "📌 Installing Istio..."
curl -L https://istio.io/downloadIstio | sh -
cd istio-*/
export PATH=$PWD/bin:$PATH
echo y | istioctl install

cd ..


#
# Will create the namespaces used by the platform (Declared in namespaces.yaml)
#
echo "📌 Creating namespaces..."
kubectl apply -f resources/namespaces.yaml

#
# Will install kubeflow pipelines from their github, this will only install the pipelines component
# of kubeflow. The version to install can be customized inside the config.yaml
#
echo "📌 Installing Kubeflow Pipelines..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINES_VERSION"
kubectl wait --for condition=established --timeout=500s crd/applications.app.k8s.io
# kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINES_VERSION"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"



echo "⏳ Waiting for kubeflow pods to be ready... (this may take a while)"
kubectl wait --for=condition=available deployment/ml-pipeline-ui -n kubeflow --timeout=500s
kubectl wait --for=condition=available deployment/ml-pipeline -n kubeflow --timeout=500s
kubectl wait --for=condition=available deployment/minio -n kubeflow --timeout=500s

#
# Minio is installed together with kubeflow pipelines, we don't need to install it ourselves.'
# It is installed under the kubeflow namespace.
# Here we will create some buckets inside of minio, this is mainly done because
# the bucket ml-models is needed for our ml-flow serving to work.
# For more information about this check the example-guide.md on the github repo. (COMMING SOON)
#

# Check if mc is installed, if not, install it temporarily.
if ! command -v mc >/dev/null 2>&1
then
    echo "🔍 mc not found, installing temporarily..."
    wget -qO mc "https://dl.min.io/client/mc/release/linux-amd64/mc"
    chmod +x mc
fi

kubectl port-forward svc/minio-service -n kubeflow 9000:9000 &
MINIO_PORT_FORWARD_PID=$!

sleep 5

mc alias set myminio http://localhost:9000 minio minio123

echo "📌 Creating Minio buckets..."

for BUCKET in $BUCKETS
do
    echo "🪣 Creating bucket: $BUCKET"
    mc mb myminio/$BUCKET || echo "⚠️ Bucket $BUCKET already exists, skipping..."
done

kill $MINIO_PORT_FORWARD_PID

#
# This will install prometheus and grafana in the platform, there is not much special
# about it. Except for that we can customize the grafana and prometheus urls and
# set our own password for Grafana.
#
echo "📌 Installing Prometheus and Grafana..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm upgrade --install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword="$GRAFANA_PASSWORD" \
  --set env.GF_SERVER_ROOT_URL="$GRAFANA_URL" \
  --set env.GF_SERVER_SERVE_FROM_SUB_PATH="true"


helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.routePrefix="/prometheus" \
  --set prometheus.prometheusSpec.externalUrl="$PROMETHEUS_URL"

echo "⏳ Waiting for Prometheus and Grafana to be ready..."
kubectl wait --for=condition=ready pods --all -n monitoring --timeout=300s
echo "✅ Grafana password has been set to '$GRAFANA_PASSWORD'!"


echo "📌 Applying Kubernetes platform resources..."
kubectl apply -f resources/manifests/

echo "⏳ Waiting for MLflow and other deployments to be ready..."
kubectl wait --for=condition=available deployment/mlflow -n mlflow --timeout=180s

#
# This will enable the istio service mesh for all our namespaces.
#
echo "📌 Activating Istio-injection..."
kubectl label namespace kubeflow istio-injection=enabled --overwrite
kubectl label namespace mlflow istio-injection=enabled --overwrite
kubectl label namespace monitoring istio-injection=enabled --overwrite

#
# Provide clear access guidance at the end of the script, improving user experience
#
echo "✅ Setup complete!"
echo ""
echo "📌 Access your MLOps Platform Services:"
echo "--------------------------------------------------"

# Run 'minikube tunnel' in a separate terminal to get the IP
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$INGRESS_HOST" ]; then
    echo "⚠️ Could not determine Ingress IP. Please run 'minikube tunnel' in a separate terminal and wait for the IP to be assigned."
    echo "Then get the IP by running 'kubectl get svc -n istio-system', the external IP is the one you need."
    export INGRESS_HOST="<External-IP>"
fi

echo "➡️  Kubeflow Pipelines: http://$INGRESS_HOST/pipeline/"
echo "➡️  MLflow Dashboard:   http://$INGRESS_HOST/mlflow/"
echo "➡️  Grafana:            http://$INGRESS_HOST/grafana/"
echo "➡️  Minio Console:      http://$INGRESS_HOST/minio/"
echo "➡️  Prometheus:         http://$INGRESS_HOST/prometheus/"

echo "--------------------------------------------------"
