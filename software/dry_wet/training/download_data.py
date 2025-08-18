import os
import argparse
import tempfile
import zipfile
from minio import Minio
from minio.error import S3Error

def download_and_extract_data(
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    object_name: str,
    output_path: str,
):
    """
    Connects to MinIO, downloads a compressed file, and extracts it to a specified directory.

    Args:
        minio_endpoint (str): The endpoint URL of the MinIO server.
        access_key (str): The access key for MinIO.
        secret_key (str): The secret key for MinIO.
        bucket_name (str): The name of the bucket containing the data.
        object_name (str): The name of the compressed file (object) in the bucket.
        output_path (str): The local path where the data should be extracted.
    """
    print("--- Data Download and Extraction Step ---")
    print(f"Configuration:")
    print(f"  MinIO Endpoint: {minio_endpoint}")
    print(f"  Bucket: {bucket_name}")
    print(f"  Object: {object_name}")
    print(f"  Output Path: {output_path}")

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

    # 2. Download the compressed file to a temporary location
    # tempfile.NamedTemporaryFile creates a file that is deleted on close.
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp_file:
        temp_zip_path = tmp_file.name
        print(f"Downloading {object_name} to temporary file: {temp_zip_path}...")
        try:
            client.fget_object(bucket_name, object_name, temp_zip_path)
            print("Download completed successfully.")
        except S3Error as e:
            print(f"Error occurred during download from MinIO: {e}")
            raise

        # 3. Extract the contents to the specified output path
        print(f"Extracting contents to {output_path}...")
        try:
            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            print(f"Extraction completed. Data is available at: {output_path}")
            print("Listing extracted files (first 5):")
            extracted_files = os.listdir(output_path)
            for file_name in extracted_files[:5]:
                print(f"  - {file_name}")
            if len(extracted_files) > 5:
                print(f"  ... and {len(extracted_files) - 5} more files.")

        except zipfile.BadZipFile:
            print(f"Error: The downloaded file '{object_name}' is not a valid zip file.")
            raise
        except Exception as e:
            print(f"An error occurred during extraction: {e}")
            raise

    print("--- Data Download and Extraction Step Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download and extract data from a MinIO bucket for model training."
    )
    
    # --- MinIO Connection Arguments ---
    parser.add_argument('--minio-endpoint', type=str, required=True, help='MinIO server endpoint URL (e.g., minio-service.kubeflow:9000).')
    parser.add_argument('--minio-access-key', type=str, required=True, help='MinIO access key.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='MinIO secret key.')
    
    # --- Data Source Arguments ---
    parser.add_argument('--bucket-name', type=str, required=True, help='Name of the source bucket in MinIO.')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the compressed data file in the bucket (e.g., road-images.zip).')
    
    # --- Output Path Argument (for Kubeflow) ---
    parser.add_argument('--output-path', type=str, required=True, help='Local directory path to extract data into. This will be provided by Kubeflow.')
    
    args = parser.parse_args()

    download_and_extract_data(
        minio_endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket_name=args.bucket_name,
        object_name=args.object_name,
        output_path=args.output_path,
    ) 