#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cloud Storage Service

This module provides a service for interacting with Google Cloud Storage.
It handles file uploads, downloads, and other GCS operations.

Key features:
- File uploads and downloads
- Signed URL generation
- Bucket and object management
- Error handling
"""

import os
import logging
import tempfile
import base64
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime, timedelta

# Import local modules
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class GCSService:
    """
    Service for interacting with Google Cloud Storage.
    
    This class provides methods for uploading, downloading, and managing files in GCS.
    """
    
    def __init__(self, bucket_name: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize the GCS service.
        
        Args:
            bucket_name: Optional bucket name. If not provided, it will be loaded from settings.
            credentials_path: Optional path to credentials file. If not provided, it will be loaded from settings.
        """
        self.bucket_name = bucket_name or settings.GCS_BUCKET_NAME
        self.credentials_path = credentials_path or settings.CREDENTIALS_PATH
        
        # Set environment variable for credentials
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        
        # Initialize the GCS client
        try:
            from google.cloud import storage
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"GCS client initialized successfully for bucket {self.bucket_name}")
        except ImportError:
            logger.error("Google Cloud Storage client library not installed. Install with: pip install google-cloud-storage")
            raise ImportError("Google Cloud Storage client library not installed")
        except Exception as e:
            logger.error(f"Error initializing GCS client: {str(e)}")
            raise
    
    def upload_file(self, source_file_path: str, destination_blob_name: str) -> Dict[str, str]:
        """
        Upload a file to GCS.
        
        Args:
            source_file_path: Path to the local file to upload
            destination_blob_name: Name of the blob to create in GCS
            
        Returns:
            Dict[str, str]: Dictionary with upload results
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(destination_blob_name)
            
            # Upload the file
            blob.upload_from_filename(source_file_path)
            
            logger.info(f"File {source_file_path} uploaded to {destination_blob_name}")
            
            # Generate a signed URL for the uploaded file
            signed_url = self.generate_signed_url(destination_blob_name)
            
            return {
                "gcs_path": destination_blob_name,
                "gcs_url": signed_url,
                "bucket": self.bucket_name
            }
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {str(e)}")
            raise
    
    def upload_from_string(self, content: str, destination_blob_name: str, content_type: str = "text/plain") -> Dict[str, str]:
        """
        Upload content as a string to GCS.
        
        Args:
            content: Content to upload
            destination_blob_name: Name of the blob to create in GCS
            content_type: Content type of the blob
            
        Returns:
            Dict[str, str]: Dictionary with upload results
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(destination_blob_name)
            
            # Upload the content
            blob.upload_from_string(content, content_type=content_type)
            
            logger.info(f"Content uploaded to {destination_blob_name}")
            
            # Generate a signed URL for the uploaded file
            signed_url = self.generate_signed_url(destination_blob_name)
            
            return {
                "gcs_path": destination_blob_name,
                "gcs_url": signed_url,
                "bucket": self.bucket_name
            }
        except Exception as e:
            logger.error(f"Error uploading content to GCS: {str(e)}")
            raise
    
    def upload_from_base64(self, base64_content: str, destination_blob_name: str, content_type: str = "application/octet-stream") -> Dict[str, str]:
        """
        Upload content from base64-encoded string to GCS.
        
        Args:
            base64_content: Base64-encoded content to upload
            destination_blob_name: Name of the blob to create in GCS
            content_type: Content type of the blob
            
        Returns:
            Dict[str, str]: Dictionary with upload results
        """
        try:
            # Decode the base64 content
            binary_content = base64.b64decode(base64_content)
            
            # Create a blob object
            blob = self.bucket.blob(destination_blob_name)
            
            # Upload the content
            blob.upload_from_string(binary_content, content_type=content_type)
            
            logger.info(f"Base64 content uploaded to {destination_blob_name}")
            
            # Generate a signed URL for the uploaded file
            signed_url = self.generate_signed_url(destination_blob_name)
            
            return {
                "gcs_path": destination_blob_name,
                "gcs_url": signed_url,
                "bucket": self.bucket_name
            }
        except Exception as e:
            logger.error(f"Error uploading base64 content to GCS: {str(e)}")
            raise
    
    def download_to_file(self, source_blob_name: str, destination_file_path: str) -> str:
        """
        Download a blob from GCS to a local file.
        
        Args:
            source_blob_name: Name of the blob to download
            destination_file_path: Path to the local file to create
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(source_blob_name)
            
            # Download the blob
            blob.download_to_filename(destination_file_path)
            
            logger.info(f"Blob {source_blob_name} downloaded to {destination_file_path}")
            
            return destination_file_path
        except Exception as e:
            logger.error(f"Error downloading blob from GCS: {str(e)}")
            raise
    
    def download_as_string(self, source_blob_name: str) -> str:
        """
        Download a blob from GCS as a string.
        
        Args:
            source_blob_name: Name of the blob to download
            
        Returns:
            str: Content of the blob as a string
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(source_blob_name)
            
            # Download the blob as a string
            content = blob.download_as_text()
            
            logger.info(f"Blob {source_blob_name} downloaded as string")
            
            return content
        except Exception as e:
            logger.error(f"Error downloading blob as string from GCS: {str(e)}")
            raise
    
    def download_as_bytes(self, source_blob_name: str) -> bytes:
        """
        Download a blob from GCS as bytes.
        
        Args:
            source_blob_name: Name of the blob to download
            
        Returns:
            bytes: Content of the blob as bytes
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(source_blob_name)
            
            # Download the blob as bytes
            content = blob.download_as_bytes()
            
            logger.info(f"Blob {source_blob_name} downloaded as bytes")
            
            return content
        except Exception as e:
            logger.error(f"Error downloading blob as bytes from GCS: {str(e)}")
            raise
    
    def list_blobs(self, prefix: Optional[str] = None) -> List[str]:
        """
        List blobs in the bucket.
        
        Args:
            prefix: Optional prefix to filter blobs
            
        Returns:
            List[str]: List of blob names
        """
        try:
            # List blobs in the bucket
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            
            # Extract blob names
            blob_names = [blob.name for blob in blobs]
            
            logger.info(f"Listed {len(blob_names)} blobs in bucket {self.bucket_name}")
            
            return blob_names
        except Exception as e:
            logger.error(f"Error listing blobs in GCS: {str(e)}")
            raise
    
    def generate_signed_url(self, blob_name: str, expiration_days: int = 7) -> str:
        """
        Generate a signed URL for a blob.
        
        Args:
            blob_name: Name of the blob
            expiration_days: Number of days until the URL expires
            
        Returns:
            str: Signed URL for the blob
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(blob_name)
            
            # Calculate expiration time
            expiration = datetime.now() + timedelta(days=expiration_days)
            
            # Generate a signed URL
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
            
            logger.info(f"Generated signed URL for blob {blob_name}")
            
            return signed_url
        except Exception as e:
            logger.error(f"Error generating signed URL for blob {blob_name}: {str(e)}")
            return ""
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from the bucket.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            bool: True if the blob was deleted, False otherwise
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(blob_name)
            
            # Delete the blob
            blob.delete()
            
            logger.info(f"Blob {blob_name} deleted from bucket {self.bucket_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting blob {blob_name} from GCS: {str(e)}")
            return False
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists in the bucket.
        
        Args:
            blob_name: Name of the blob to check
            
        Returns:
            bool: True if the blob exists, False otherwise
        """
        try:
            # Create a blob object
            blob = self.bucket.blob(blob_name)
            
            # Check if the blob exists
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if blob {blob_name} exists in GCS: {str(e)}")
            return False
