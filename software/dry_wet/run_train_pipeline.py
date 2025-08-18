import kfp
import argparse
from datetime import datetime
from pipeline import create_dry_wet_training_pipeline

def run_kubeflow_pipeline(
    kubeflow_host: str,
    pipeline_arguments: dict,
    training_image: str,
    compiled_pipeline_path: str = "pipeline.yaml"
):
    """
    Compiles the pipeline, connects to a Kubeflow instance, and creates a run.

    Args:
        kubeflow_host (str): The URL of the Kubeflow Pipelines API endpoint.
        pipeline_arguments (dict): A dictionary of arguments to pass to the pipeline run.
        training_image (str): The Docker image to use for training pipeline components.
        compiled_pipeline_path (str): The path to save the compiled YAML file.
    """
    print("--- Starting Kubeflow Pipeline Run ---")
    
    # 1. Compile the pipeline
    # This converts the Python DSL into a static YAML file that Kubeflow can read.
    # Create pipeline with the specified training image, then compile it.
    print(f"Creating pipeline with training image: {training_image}")
    pipeline_func = create_dry_wet_training_pipeline(training_image)
    
    print(f"Compiling pipeline to {compiled_pipeline_path}...")
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=compiled_pipeline_path,
    )
    print("Compilation successful.")

    # 2. Initialize the KFP client
    # The host should point to your Kubeflow Pipelines API service.
    # If running in-cluster, you might not need to specify it.
    # For local execution, you may need to set up port-forwarding.
    print(f"Connecting to Kubeflow Pipelines at: {kubeflow_host}")
    try:
        client = kfp.Client(host=kubeflow_host)
        print("Successfully connected to Kubeflow client.")
    except Exception as e:
        print(f"Failed to connect to Kubeflow client. Ensure the host URL is correct and accessible. Error: {e}")
        return

    # 3. Create or get experiment
    experiment_name = 'Dry-Wet Model Training Experiment'
    print(f"Creating or getting experiment: {experiment_name}")
    try:
        # Try to get existing experiment first
        try:
            experiment = client.get_experiment(experiment_name=experiment_name)
            print(f"Using existing experiment: {experiment.experiment_id}")
        except:
            # Create new experiment if it doesn't exist
            experiment = client.create_experiment(name=experiment_name)
            print(f"Created new experiment: {experiment.experiment_id}")
    except Exception as e:
        print(f"Failed to create or get experiment. Error: {e}")
        return

    # 4. Create a new run
    # Generate a custom run name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    run_name = f"dry-wet-training {timestamp}"
    
    # The client uploads the compiled YAML and starts the pipeline with your arguments.
    print("Creating a new pipeline run...")
    print(f"Run name: {run_name}")
    print(f"With arguments: {pipeline_arguments}")
    try:
        run_result = client.create_run_from_pipeline_package(
            compiled_pipeline_path,
            arguments=pipeline_arguments,
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        )
        print("Successfully created a run. You can view it in the Kubeflow UI.")
        print(f"Run Details URL: {kubeflow_host}/#/runs/details/{run_result.run_id}")
    except Exception as e:
        print(f"Failed to create a run. Check your connection and permissions. Error: {e}")

    print("--- Kubeflow Pipeline Run Submission Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compile and run the Dry-Wet training pipeline on Kubeflow.")

    # --- Kubeflow Connection ---
    parser.add_argument(
        '--kubeflow-host', 
        type=str, 
        default="http://localhost/pipeline",  # Istio Ingress Gateway IP
        help="The host address of the Kubeflow Pipelines API (e.g., http://<kfp-ip>:<port>)."
    )
    
    # --- Training Image ---
    parser.add_argument(
        '--training-image', 
        type=str, 
        default="dry_wet_train:v1.0.0", 
        help="Docker image for training pipeline components (e.g., dry_wet_train:v1.0.0)."
    )
    
    # --- MinIO Connection Arguments ---
    parser.add_argument('--minio-endpoint', type=str, default="minio-service.kubeflow:9000", help="MinIO server endpoint URL.")
    parser.add_argument('--minio-access-key', type=str, default="minio", help="MinIO access key.")
    parser.add_argument('--minio-secret-key', type=str, default="minio123", help="MinIO secret key.")

    # --- Data Source Arguments ---
    parser.add_argument('--source-data-bucket', type=str, default="ml-data", help="Name of the MinIO bucket containing the source data.")
    parser.add_argument('--source-data-object', type=str, default="dry-wet-data.zip", help="Name of the source data zip file in the bucket.")
    
    # --- Model Destination Arguments ---
    parser.add_argument('--destination-model-bucket', type=str, default="ml-models", help="Name of the MinIO bucket to store the trained model.")
    parser.add_argument('--destination-model-name', type=str, default="dry-wet-road-classifier", help="Name for the destination folder in MinIO for the model.")

    args = parser.parse_args()
    
    # Package the pipeline arguments into a dictionary
    arguments_for_pipeline = {
        'minio_endpoint': args.minio_endpoint,
        'minio_access_key': args.minio_access_key,
        'minio_secret_key': args.minio_secret_key,
        'source_data_bucket': args.source_data_bucket,
        'source_data_object': args.source_data_object,
        'destination_model_bucket': args.destination_model_bucket,
        'destination_model_name': args.destination_model_name,
    }

    run_kubeflow_pipeline(
        kubeflow_host=args.kubeflow_host,
        pipeline_arguments=arguments_for_pipeline,
        training_image=args.training_image,
    ) 