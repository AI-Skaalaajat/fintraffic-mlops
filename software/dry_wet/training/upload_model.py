import os
import argparse
from datetime import datetime
from minio import Minio
from minio.error import S3Error

def upload_artifacts_to_minio(
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    model_input_path: str,
    destination_path: str = ""
):
    """
    Uploads all files from a local directory (KFP artifact path) to a MinIO bucket.

    Args:
        minio_endpoint (str): The endpoint URL of the MinIO server.
        access_key (str): The access key for MinIO.
        secret_key (str): The secret key for MinIO.
        bucket_name (str): The name of the destination bucket in MinIO.
        model_input_path (str): The local directory containing model artifacts to upload.
        destination_path (str): A path/prefix to prepend to the object names in the bucket.
    """
    print("--- Model Upload Step ---")
    
    # Create a versioned folder name using the destination_path as a base and appending the current date
    date_suffix = datetime.now().strftime('%Y-%m-%d')
    versioned_folder_name = f"{destination_path}-{date_suffix}" if destination_path else f"model-{date_suffix}"
    
    print(f"Configuration:")
    print(f"  MinIO Endpoint: {minio_endpoint}")
    print(f"  Destination Bucket: {bucket_name}")
    print(f"  Destination Path: {versioned_folder_name}")
    print(f"  Artifacts Source: {model_input_path}")

    # 1. Initialize MinIO client
    try:
        client = Minio(
            minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # Set to True if using HTTPS
        )
        print("Successfully initialized MinIO client.")
    except Exception as e:
        print(f"Error initializing MinIO client: {e}")
        raise
        
    # 2. Ensure the destination bucket exists
    try:
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' did not exist. Created it.")
        else:
            print(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        print(f"Error checking or creating bucket: {e}")
        raise

    # 3. Walk through the input directory and upload each file
    try:
        print("Starting artifact upload...")
        for root, _, files in os.walk(model_input_path):
            if not files:
                continue
            for filename in files:
                local_file_path = os.path.join(root, filename)
                
                # Construct the object name for MinIO, including the versioned destination path
                relative_path = os.path.relpath(local_file_path, model_input_path)
                object_name = os.path.join(versioned_folder_name, relative_path)
                
                print(f"  Uploading '{local_file_path}' to '{bucket_name}/{object_name}'...")
                
                client.fput_object(
                    bucket_name,
                    object_name,
                    local_file_path,
                )
        print("All artifacts uploaded successfully.")
        
    except FileNotFoundError:
        print(f"Error: The specified input path '{model_input_path}' does not exist.")
        raise
    except S3Error as e:
        print(f"An error occurred during upload to MinIO: {e}")
        raise

    print("--- Model Upload Step Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Upload model artifacts from a Kubeflow pipeline step to a MinIO bucket."
    )
    
    # --- MinIO Connection Arguments ---
    parser.add_argument('--minio-endpoint', type=str, required=True, help='MinIO server endpoint URL.')
    parser.add_argument('--minio-access-key', type=str, required=True, help='MinIO access key.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='MinIO secret key.')
    
    # --- Destination Arguments ---
    parser.add_argument('--bucket-name', type=str, required=True, help='Name of the destination bucket in MinIO.')
    parser.add_argument('--destination-path', type=str, default="", help='Optional path/prefix to store artifacts under within the bucket (e.g., "my-experiment/run-123").')

    # --- Input Path Argument (from Kubeflow) ---
    parser.add_argument('--model-input-path', type=str, required=True, help='Local directory path where model artifacts are stored. This is provided by the previous KFP step.')
    
    args = parser.parse_args()

    upload_artifacts_to_minio(
        minio_endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket_name=args.bucket_name,
        model_input_path=args.model_input_path,
        destination_path=args.destination_path,
    ) 