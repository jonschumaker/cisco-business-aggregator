#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Storage Utilities

This module provides utility functions for storage operations.
It handles operations like saving files, reading files, and managing storage.

Key features:
- Local and GCS file storage
- File reading and writing
- Directory management
- URL generation
"""

import os
import logging
import tempfile
import shutil
import base64
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Import local modules
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

def save_file(content: Union[str, bytes], file_path: str, use_gcs: bool = False, 
             content_type: Optional[str] = None) -> Dict[str, str]:
    """
    Save content to a file, either locally or in GCS.
    
    Args:
        content: Content to save
        file_path: Path to save the file
        use_gcs: Whether to use GCS or local storage
        content_type: Content type for GCS storage
        
    Returns:
        Dict[str, str]: Dictionary with file information
    """
    try:
        # Determine if we should use GCS based on the parameter and settings
        use_gcs_storage = use_gcs and not settings.USE_LOCAL_STORAGE
        
        if use_gcs_storage:
            # Save to GCS
            return save_to_gcs(content, file_path, content_type)
        else:
            # Save locally
            return save_locally(content, file_path)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return {"error": str(e)}

def save_locally(content: Union[str, bytes], file_path: str) -> Dict[str, str]:
    """
    Save content to a local file.
    
    Args:
        content: Content to save
        file_path: Path to save the file
        
    Returns:
        Dict[str, str]: Dictionary with file information
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine if content is binary or text
        is_binary = isinstance(content, bytes)
        
        # Save the file
        if is_binary:
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"File saved locally: {file_path}")
        
        return {
            "local_path": file_path,
            "is_binary": is_binary
        }
    except Exception as e:
        logger.error(f"Error saving file locally: {str(e)}")
        return {"error": str(e)}

def save_to_gcs(content: Union[str, bytes], file_path: str, 
               content_type: Optional[str] = None) -> Dict[str, str]:
    """
    Save content to a file in GCS.
    
    Args:
        content: Content to save
        file_path: Path to save the file in GCS
        content_type: Content type for GCS storage
        
    Returns:
        Dict[str, str]: Dictionary with file information
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Determine content type if not provided
        if not content_type:
            if file_path.endswith('.md'):
                content_type = 'text/markdown'
            elif file_path.endswith('.json'):
                content_type = 'application/json'
            elif file_path.endswith('.docx'):
                content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            else:
                content_type = 'text/plain'
        
        # Check if content is base64-encoded
        if isinstance(content, str) and content.startswith('data:') and ';base64,' in content:
            # Extract base64 content
            content = content.split(';base64,')[1]
            
            # Upload base64 content
            result = gcs_service.upload_from_base64(content, file_path, content_type)
        elif isinstance(content, str):
            # Upload string content
            result = gcs_service.upload_from_string(content, file_path, content_type)
        else:
            # Upload binary content
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Upload the temporary file
            result = gcs_service.upload_file(temp_file_path, file_path)
            
            # Delete the temporary file
            os.unlink(temp_file_path)
        
        logger.info(f"File saved to GCS: {file_path}")
        
        return result
    except Exception as e:
        logger.error(f"Error saving file to GCS: {str(e)}")
        return {"error": str(e)}

def read_file(file_path: str, use_gcs: bool = False) -> Union[str, bytes, None]:
    """
    Read content from a file, either locally or from GCS.
    
    Args:
        file_path: Path to the file
        use_gcs: Whether to use GCS or local storage
        
    Returns:
        Union[str, bytes, None]: File content or None if the file doesn't exist
    """
    try:
        # Determine if we should use GCS based on the parameter and settings
        use_gcs_storage = use_gcs and not settings.USE_LOCAL_STORAGE
        
        if use_gcs_storage:
            # Read from GCS
            return read_from_gcs(file_path)
        else:
            # Read locally
            return read_locally(file_path)
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return None

def read_locally(file_path: str) -> Union[str, bytes, None]:
    """
    Read content from a local file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Union[str, bytes, None]: File content or None if the file doesn't exist
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        # Determine if the file is binary
        is_binary = is_binary_file(file_path)
        
        # Read the file
        if is_binary:
            with open(file_path, 'rb') as f:
                content = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        logger.info(f"File read locally: {file_path}")
        
        return content
    except Exception as e:
        logger.error(f"Error reading file locally: {str(e)}")
        return None

def read_from_gcs(file_path: str) -> Union[str, bytes, None]:
    """
    Read content from a file in GCS.
    
    Args:
        file_path: Path to the file in GCS
        
    Returns:
        Union[str, bytes, None]: File content or None if the file doesn't exist
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Check if the file exists
        if not gcs_service.blob_exists(file_path):
            logger.warning(f"File not found in GCS: {file_path}")
            return None
        
        # Determine if the file is binary
        is_binary = file_path.endswith(('.docx', '.xlsx', '.pdf', '.png', '.jpg', '.jpeg', '.gif'))
        
        # Read the file
        if is_binary:
            content = gcs_service.download_as_bytes(file_path)
        else:
            content = gcs_service.download_as_string(file_path)
        
        logger.info(f"File read from GCS: {file_path}")
        
        return content
    except Exception as e:
        logger.error(f"Error reading file from GCS: {str(e)}")
        return None

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is binary, False otherwise
    """
    # Check file extension
    binary_extensions = ['.docx', '.xlsx', '.pdf', '.png', '.jpg', '.jpeg', '.gif']
    if any(file_path.endswith(ext) for ext in binary_extensions):
        return True
    
    # Check file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return False
    except UnicodeDecodeError:
        return True

def list_files_in_gcs(prefix: Optional[str] = None) -> List[str]:
    """
    List files in GCS.
    
    Args:
        prefix: Optional prefix to filter files
        
    Returns:
        List[str]: List of file paths
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # List files
        return gcs_service.list_blobs(prefix)
    except Exception as e:
        logger.error(f"Error listing files in GCS: {str(e)}")
        return []

def generate_signed_url(file_path: str, expiration_days: int = 7) -> Optional[str]:
    """
    Generate a signed URL for a file in GCS.
    
    Args:
        file_path: Path to the file in GCS
        expiration_days: Number of days until the URL expires
        
    Returns:
        Optional[str]: Signed URL or None if an error occurs
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Generate signed URL
        return gcs_service.generate_signed_url(file_path, expiration_days)
    except Exception as e:
        logger.error(f"Error generating signed URL: {str(e)}")
        return None

def upload_to_gcs(local_path: str, gcs_path: str) -> Dict[str, str]:
    """
    Upload a local file to GCS.
    
    Args:
        local_path: Path to the local file
        gcs_path: Path to save the file in GCS
        
    Returns:
        Dict[str, str]: Dictionary with upload results
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Upload the file
        return gcs_service.upload_file(local_path, gcs_path)
    except Exception as e:
        logger.error(f"Error uploading file to GCS: {str(e)}")
        return {"error": str(e)}

def download_from_gcs(gcs_path: str, local_path: str) -> str:
    """
    Download a file from GCS to a local path.
    
    Args:
        gcs_path: Path to the file in GCS
        local_path: Path to save the file locally
        
    Returns:
        str: Path to the downloaded file
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Download the file
        return gcs_service.download_to_file(gcs_path, local_path)
    except Exception as e:
        logger.error(f"Error downloading file from GCS: {str(e)}")
        raise

def download_blob_as_string(gcs_path: str) -> Optional[str]:
    """
    Download a blob from GCS as a string.
    
    Args:
        gcs_path: Path to the blob in GCS
        
    Returns:
        Optional[str]: Blob content as a string or None if an error occurs
    """
    try:
        # Import GCS service
        from services.gcs_service import GCSService
        
        # Initialize GCS service
        gcs_service = GCSService()
        
        # Download the blob
        return gcs_service.download_as_string(gcs_path)
    except Exception as e:
        logger.error(f"Error downloading blob as string: {str(e)}")
        return None

def get_temp_dir() -> str:
    """
    Get a temporary directory.
    
    Returns:
        str: Path to the temporary directory
    """
    return tempfile.mkdtemp()

def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure that a directory exists.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def extract_company_name(url: str, customer_name: Optional[str] = None) -> str:
    """
    Extract company name from URL or customer name.
    
    Args:
        url: URL to extract company name from
        customer_name: Optional customer name to extract company name from
        
    Returns:
        str: Extracted company name
    """
    # Try to extract from customer name first
    if customer_name:
        # If customer name contains parentheses, extract the part before them
        if "(" in customer_name:
            return customer_name.split("(")[0].strip()
        else:
            return customer_name
    
    # Extract from URL
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Get the first part of the domain (before the first dot)
        company = domain.split('.')[0]
        
        # Clean up and capitalize
        company = company.replace('-', ' ').replace('_', ' ')
        company = ' '.join(word.capitalize() for word in company.split())
        
        return company
    except:
        # Fallback to a generic name
        return "Company"
