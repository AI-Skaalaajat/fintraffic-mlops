from kfp import dsl
from kfp.dsl import Input, Output, Dataset

# The single, unified image for all pipeline steps will be specified at compilation time.
# In a production environment, this should be a versioned image from a container registry.

# ==============================================================================
# Pipeline Factory Function
# ==============================================================================

def create_dry_wet_training_pipeline(training_image: str = "dry_wet_train:v1.0.0"):
    """Factory function to create a pipeline with a specific training image."""
    
    @dsl.container_component
    def download_data_component(
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str,
        object_name: str,
        data_path: Output[Dataset],
    ):
        """Downloads and extracts data from MinIO."""
        return dsl.ContainerSpec(
            image=training_image,
            command=["python", "download_data.py"],
            args=[
                "--minio-endpoint", minio_endpoint,
                "--minio-access-key", minio_access_key,
                "--minio-secret-key", minio_secret_key,
                "--bucket-name", bucket_name,
                "--object-name", object_name,
                "--output-path", data_path.path,
            ],
        )

    @dsl.container_component
    def train_model_component(
        data_path: Input[Dataset],
        model_output_path: Output[Dataset],
    ):
        """Trains the model using the prepared data."""
        return dsl.ContainerSpec(
            image=training_image,
            command=["python", "train.py"],
            args=[
                "--config-path", "config.yaml", 
                "--data-path", data_path.path,
                "--model-output-path", model_output_path.path,
            ],
        )

    @dsl.container_component
    def upload_model_component(
        model_input_path: Input[Dataset],
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str,
        destination_path: str,
    ):
        """Uploads the trained model artifacts to MinIO."""
        return dsl.ContainerSpec(
            image=training_image,
            command=["python", "upload_model.py"],
            args=[
                "--minio-endpoint", minio_endpoint,
                "--minio-access-key", minio_access_key,
                "--minio-secret-key", minio_secret_key,
                "--bucket-name", bucket_name,
                "--model-input-path", model_input_path.path,
                "--destination-path", destination_path,
            ],
        )

    @dsl.pipeline(
        name='Dry-Wet Model Training Pipeline',
        description='A pipeline that downloads data, trains a model, and uploads the artifacts.'
    )
    def dry_wet_training_pipeline(
        # MinIO connection details.
        minio_endpoint: str = "minio-service.kubeflow:9000",
        minio_access_key: str = "minio",
        minio_secret_key: str = "minio123",
        # Data source parameters
        source_data_bucket: str = "ml-data",
        source_data_object: str = "dry-wet-data.zip",
        # Model destination parameters
        destination_model_bucket: str = "ml-models",
        destination_model_name: str = "dry-wet-road-classifier"
    ):
        """Defines the Kubeflow pipeline structure by connecting the components."""

        # Step 1: Download data
        download_task = download_data_component(
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            bucket_name=source_data_bucket,
            object_name=source_data_object,
        )

        # Step 2: Train the model, using the output from the download step
        train_task = train_model_component(
            data_path=download_task.outputs["data_path"]
        )
        train_task.set_env_variable(name="MLFLOW_TRACKING_URI", value="http://mlflow-service.mlflow.svc.cluster.local:5000")
        train_task.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio-service.kubeflow.svc.cluster.local:9000")
        train_task.set_env_variable(name="AWS_ACCESS_KEY_ID", value="minio")
        train_task.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value="minio123")

        # Step 3: Upload the trained model artifacts, using the output from the train step
        upload_task = upload_model_component(
            model_input_path=train_task.outputs["model_output_path"],
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            bucket_name=destination_model_bucket,
            destination_path=destination_model_name,
        )

    return dry_wet_training_pipeline 