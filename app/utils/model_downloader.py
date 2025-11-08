"""
Utility to download model files on startup using Azure Blob Storage
"""
import os
from azure.storage.blob import BlobServiceClient
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Hardcoded model configuration
MODEL_CONTAINER = "model-files"
GAIT_MODEL_BLOB = "GaitBase_DA-60000.pt"
YOLO_MODEL_BLOB = "yolov8n.pt"
YOLO_MODEL_PATH = "yolov8n.pt"


def download_blob_to_file(blob_service_client: BlobServiceClient, 
                          container_name: str, blob_name: str, 
                          destination: str) -> bool:
    """
    Download a blob from Azure Storage to local file
    
    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Container name in Azure Storage
        blob_name: Blob name/path in container
        destination: Local path to save file
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        dest_dir = os.path.dirname(destination)
        if dest_dir:  # Only create if there's a directory part
            os.makedirs(dest_dir, exist_ok=True)
        
        logger.info(f"Downloading {container_name}/{blob_name} to {destination}...")
        print(f"  Downloading from Azure: {container_name}/{blob_name}")
        
        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        # Download blob
        with open(destination, "wb") as download_file:
            blob_data = blob_client.download_blob()
            download_file.write(blob_data.readall())
        
        file_size = os.path.getsize(destination)
        logger.info(f"✓ Successfully downloaded to {destination} ({file_size / 1024 / 1024:.2f} MB)")
        print(f"  ✓ Download complete ({file_size / 1024 / 1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {container_name}/{blob_name}: {e}")
        print(f"  ✗ Download failed: {e}")
        return False


def ensure_models_exist(connection_string: str, gait_model_path: str) -> None:
    """
    Ensure model files exist, download from Azure if missing
    
    Args:
        connection_string: Azure Storage connection string
        gait_model_path: Path to GaitBase model
    """
    print("\n" + "="*60)
    print("CHECKING MODEL FILES")
    print("="*60)
    
    try:
        # Initialize Azure Blob Service Client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Check GaitBase model
        if os.path.exists(gait_model_path):
            file_size = os.path.getsize(gait_model_path)
            logger.info(f"✓ GaitBase model found: {gait_model_path}")
            print(f"✓ GaitBase model found: {gait_model_path} ({file_size / 1024 / 1024:.2f} MB)")
        else:
            logger.warning(f"✗ GaitBase model not found: {gait_model_path}")
            print(f"✗ GaitBase model not found at {gait_model_path}")
            
            if download_blob_to_file(blob_service_client, MODEL_CONTAINER, 
                                    GAIT_MODEL_BLOB, gait_model_path):
                print(f"✓ GaitBase model downloaded successfully")
            else:
                logger.error(f"Failed to download GaitBase model. Gait recognition may not work properly.")
                print(f"⚠️  WARNING: Failed to download GaitBase model")
                print(f"   Gait recognition will use fallback feature extraction")
        
        # Check YOLO model
        if os.path.exists(YOLO_MODEL_PATH):
            file_size = os.path.getsize(YOLO_MODEL_PATH)
            logger.info(f"✓ YOLO model found: {YOLO_MODEL_PATH}")
            print(f"✓ YOLO model found: {YOLO_MODEL_PATH} ({file_size / 1024 / 1024:.2f} MB)")
        else:
            logger.warning(f"✗ YOLO model not found: {YOLO_MODEL_PATH}")
            print(f"✗ YOLO model not found at {YOLO_MODEL_PATH}")
            
            if download_blob_to_file(blob_service_client, MODEL_CONTAINER,
                                    YOLO_MODEL_BLOB, YOLO_MODEL_PATH):
                print(f"✓ YOLO model downloaded successfully")
            else:
                logger.error(f"Failed to download YOLO model. Person detection may not work.")
                print(f"⚠️  WARNING: Failed to download YOLO model")
                print(f"   Person detection will not work without this model")
        
    except Exception as e:
        logger.error(f"Error initializing Azure Blob Storage: {e}")
        print(f"⚠️  ERROR: Could not connect to Azure Storage: {e}")
        print(f"   Model download skipped. Ensure AZURE_STORAGE_CONNECTION_STRING is set correctly.")
    
    print("="*60)
    print("MODEL CHECK COMPLETE")
    print("="*60 + "\n")

