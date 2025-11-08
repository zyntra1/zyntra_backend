"""
Video storage utilities for Azure Blob Storage
"""
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import UploadFile
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class AzureBlobStorage:
    """Handle video file storage in Azure Blob Storage"""
    
    def __init__(self):
        """Initialize Azure Blob Storage client"""
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.AZURE_STORAGE_CONNECTION_STRING
            )
            self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME
            self._ensure_container_exists()
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.warning(f"Container check/creation warning: {e}")
    
    async def save_user_video(self, file: UploadFile, user_id: int) -> str:
        """
        Save user's gait video to Azure Blob Storage
        
        Args:
            file: Uploaded video file
            user_id: ID of the user
            
        Returns:
            Blob name (path in container)
        """
        try:
            # Generate unique blob name
            file_extension = os.path.splitext(file.filename)[1]
            blob_name = f"user_videos/user_{user_id}_{uuid.uuid4().hex}{file_extension}"
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Upload file
            content = await file.read()
            blob_client.upload_blob(content, overwrite=True)
            
            logger.info(f"Uploaded user video: {blob_name}")
            return blob_name
        except Exception as e:
            logger.error(f"Failed to upload user video: {e}")
            raise
    
    async def save_cctv_video(self, file: UploadFile, admin_id: int) -> str:
        """
        Save admin's CCTV video to Azure Blob Storage
        
        Args:
            file: Uploaded video file
            admin_id: ID of the admin
            
        Returns:
            Blob name (path in container)
        """
        try:
            # Generate unique blob name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(file.filename)[1]
            blob_name = f"cctv_videos/cctv_admin_{admin_id}_{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Upload file
            content = await file.read()
            blob_client.upload_blob(content, overwrite=True)
            
            logger.info(f"Uploaded CCTV video: {blob_name}")
            return blob_name
        except Exception as e:
            logger.error(f"Failed to upload CCTV video: {e}")
            raise
    
    def get_processed_video_path(self, log_id: int) -> str:
        """
        Generate blob name for processed video
        
        Args:
            log_id: Recognition log ID
            
        Returns:
            Blob name for processed video
        """
        return f"processed_videos/processed_log_{log_id}.mp4"
    
    def upload_processed_video(self, local_file_path: str, log_id: int) -> str:
        """
        Upload processed video from local file to Azure Blob Storage
        
        Args:
            local_file_path: Path to local processed video file
            log_id: Recognition log ID
            
        Returns:
            Blob name of uploaded processed video
        """
        try:
            blob_name = self.get_processed_video_path(log_id)
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Upload file
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Uploaded processed video: {blob_name}")
            
            # Clean up local file
            try:
                os.remove(local_file_path)
                logger.info(f"Removed local processed video: {local_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove local file {local_file_path}: {e}")
            
            return blob_name
        except Exception as e:
            logger.error(f"Failed to upload processed video: {e}")
            raise
    
    def download_to_local(self, blob_name: str, local_path: str) -> str:
        """
        Download blob to local file
        
        Args:
            blob_name: Name of blob to download
            local_path: Local path to save file
            
        Returns:
            Path to downloaded file
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download blob
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            logger.info(f"Downloaded blob {blob_name} to {local_path}")
            return local_path
        except ResourceNotFoundError:
            logger.error(f"Blob not found: {blob_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to download blob {blob_name}: {e}")
            raise
    
    def generate_download_url(self, blob_name: str, expiry_hours: int = 1) -> Optional[str]:
        """
        Generate SAS URL for downloading blob
        
        Args:
            blob_name: Name of blob
            expiry_hours: Hours until URL expires (default: 1)
            
        Returns:
            SAS URL for downloading blob, or None if blob doesn't exist
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Check if blob exists
            if not blob_client.exists():
                logger.warning(f"Blob does not exist: {blob_name}")
                return None
            
            # Extract account info from connection string
            conn_str_parts = {
                part.split('=', 1)[0]: part.split('=', 1)[1]
                for part in settings.AZURE_STORAGE_CONNECTION_STRING.split(';')
                if '=' in part
            }
            account_name = conn_str_parts.get('AccountName')
            account_key = conn_str_parts.get('AccountKey')
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )
            
            # Construct URL
            url = f"{blob_client.url}?{sas_token}"
            logger.info(f"Generated SAS URL for blob: {blob_name}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate SAS URL for {blob_name}: {e}")
            return None
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if blob exists
        
        Args:
            blob_name: Name of blob to check
            
        Returns:
            True if blob exists
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Error checking blob existence {blob_name}: {e}")
            return False
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob
        
        Args:
            blob_name: Name of blob to delete
            
        Returns:
            True if blob was deleted successfully
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Error deleting blob {blob_name}: {e}")
            return False
    
    # Legacy compatibility methods (for old interface)
    def get_absolute_path(self, blob_name: str) -> str:
        """Legacy method - returns blob name as-is"""
        return blob_name
    
    def file_exists(self, blob_name: str) -> bool:
        """Legacy method - checks if blob exists"""
        return self.blob_exists(blob_name)
    
    def delete_file(self, blob_name: str) -> bool:
        """Legacy method - deletes blob"""
        return self.delete_blob(blob_name)


# Singleton instance
video_storage = AzureBlobStorage()

