#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report Utilities

This module provides utilities for report generation, management, and processing.
It handles operations like saving reports in different formats, checking for existing
reports, and generating report filenames.

Key features:
- Report saving in multiple formats (Markdown, Word, JSON)
- Report metadata extraction and management
- Report filename generation
- Existing report checking
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import local modules
from utils.file_utils import (
    create_markdown_report, 
    create_error_report,
    create_placeholder_report,
    markdown_to_word, 
    markdown_to_json,
    extract_company_name
)
from utils.storage import (
    save_file,
    list_files_in_gcs,
    generate_signed_url,
    ensure_directory_exists
)

# Configure logging
logger = logging.getLogger(__name__)

# ===== Report Configuration =====

# Get local reports directory from environment variable or use default
LOCAL_REPORTS_DIR = os.getenv("LOCAL_REPORTS_DIR", "reports")

# Check if we should use local storage instead of GCS
USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "true").lower() in ["true", "1", "yes"]

# GCS folder for reports
GCS_FOLDER = "news-reports"

# ===== Report Generation Functions =====

def save_markdown_report(
    url: str, 
    content: str, 
    topic: str, 
    customer_name: str = None, 
    customer_metadata: dict = None
) -> Dict[str, str]:
    """
    Save the research report as a Markdown file and convert to Word document and JSON.
    
    Args:
        url: The URL associated with the report
        content: The main content of the report
        topic: The topic of the report
        customer_name: Optional customer name
        customer_metadata: Optional dictionary of customer metadata
        
    Returns:
        Dict[str, str]: Dictionary with file paths/URLs
    """
    try:
        # Create a filename based on the customer name or URL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if customer_name:
            # Use customer name from SAVM_NAME_WITH_ID
            # Replace any characters that might be invalid in filenames
            safe_name = "".join([c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in customer_name])
            base_filename = f"{safe_name}_{timestamp}"
        else:
            # Fallback to URL if customer name not provided
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]  # Remove 'www.' prefix
            base_filename = f"OUTCOMES_{domain}_{timestamp}"
        
        # Create markdown content with proper formatting
        markdown_content = create_markdown_report(url, content, topic, customer_name, customer_metadata)
        
        # Determine if we're using local storage or GCS
        if USE_LOCAL_STORAGE:
            # Create local reports directory if it doesn't exist
            ensure_directory_exists(LOCAL_REPORTS_DIR)
            
            # Create company-specific subdirectory
            company_name = extract_company_name(url, customer_name)
            safe_company_name = "".join([c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in company_name])
            company_dir = os.path.join(LOCAL_REPORTS_DIR, safe_company_name)
            ensure_directory_exists(company_dir)
            
            # Save the markdown file locally
            md_file_path = os.path.join(company_dir, f"{base_filename}.md")
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown report saved locally: {md_file_path}")
            
            # Save the JSON version with chunked sections
            json_file_path = os.path.join(company_dir, f"{base_filename}.json")
            json_data = markdown_to_json(markdown_content, url, topic, customer_name, company_name, customer_metadata)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report saved locally: {json_file_path}")
            
            # Convert to Word document
            docx_file_path = os.path.join(company_dir, f"{base_filename}.docx")
            try:
                markdown_to_word(markdown_content, docx_file_path, customer_name or url)
                logger.info(f"Word document saved locally: {docx_file_path}")
            except Exception as e:
                logger.error(f"Error converting to Word: {str(e)}")
                docx_file_path = None
            
            # Return local file paths
            result = {}
            if os.path.exists(md_file_path):
                result["markdown_local_path"] = md_file_path
            if os.path.exists(json_file_path):
                result["json_local_path"] = json_file_path
            if docx_file_path and os.path.exists(docx_file_path):
                result["docx_local_path"] = docx_file_path
            
            if not result:
                logger.error("No files were successfully saved locally")
                return None
                
            return result
        else:
            # Use GCS storage
            result = {}
            
            # Save markdown file to GCS
            md_gcs_path = f"{GCS_FOLDER}/{base_filename}.md"
            md_result = save_file(markdown_content, md_gcs_path, use_gcs=True)
            if "gcs_url" in md_result:
                result["markdown_gcs_url"] = md_result["gcs_url"]
                result["markdown_gcs_path"] = md_result["gcs_path"]
            
            # Save JSON file to GCS
            json_data = markdown_to_json(markdown_content, url, topic, customer_name, None, customer_metadata)
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            json_gcs_path = f"{GCS_FOLDER}/{base_filename}.json"
            json_result = save_file(json_content, json_gcs_path, use_gcs=True)
            if "gcs_url" in json_result:
                result["json_gcs_url"] = json_result["gcs_url"]
                result["json_gcs_path"] = json_result["gcs_path"]
            
            # Create Word document in a temporary file, then upload to GCS
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(prefix="report_")
            try:
                docx_temp_path = os.path.join(temp_dir, f"{base_filename}.docx")
                try:
                    markdown_to_word(markdown_content, docx_temp_path, customer_name or url)
                    
                    # Upload Word document to GCS
                    docx_gcs_path = f"{GCS_FOLDER}/{base_filename}.docx"
                    with open(docx_temp_path, 'rb') as f:
                        docx_content = f.read()
                    
                    # Convert binary content to base64 for storage
                    import base64
                    docx_b64 = base64.b64encode(docx_content).decode('utf-8')
                    
                    # Save as binary file
                    docx_result = save_file(docx_b64, docx_gcs_path, use_gcs=True)
                    if "gcs_url" in docx_result:
                        result["docx_gcs_url"] = docx_result["gcs_url"]
                        result["docx_gcs_path"] = docx_result["gcs_path"]
                except Exception as e:
                    logger.error(f"Error creating Word document: {str(e)}")
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
            
            if not result:
                logger.error("No files were successfully saved to GCS")
                return None
                
            return result
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return None

def save_error_report(
    url: str, 
    error_message: str, 
    customer_name: str = None, 
    customer_metadata: dict = None
) -> Dict[str, str]:
    """
    Save an error report when research fails.
    
    Args:
        url: The URL that was being researched
        error_message: The error message
        customer_name: Optional customer name
        customer_metadata: Optional dictionary of customer metadata
        
    Returns:
        Dict[str, str]: Dictionary with file paths/URLs
    """
    try:
        # Extract company name
        company_name = extract_company_name(url, customer_name)
        
        # Create error report content
        error_report = create_error_report(url, error_message, company_name, customer_name)
        
        # Save the report using the existing function
        return save_markdown_report(
            url=url,
            content=error_report,
            topic=f"Error Report for {company_name}",
            customer_name=customer_name,
            customer_metadata=customer_metadata
        )
    except Exception as e:
        logger.error(f"Error saving error report: {str(e)}")
        return None

def save_placeholder_report(
    url: str, 
    lookback_days: int, 
    customer_name: str = None, 
    customer_metadata: dict = None
) -> Dict[str, str]:
    """
    Save a placeholder report when no significant news is found.
    
    Args:
        url: The URL that was being researched
        lookback_days: The number of days that were searched
        customer_name: Optional customer name
        customer_metadata: Optional dictionary of customer metadata
        
    Returns:
        Dict[str, str]: Dictionary with file paths/URLs
    """
    try:
        # Extract company name
        company_name = extract_company_name(url, customer_name)
        
        # Create placeholder report content
        placeholder_report = create_placeholder_report(url, lookback_days, company_name, customer_name)
        
        # Save the report using the existing function
        return save_markdown_report(
            url=url,
            content=placeholder_report,
            topic=f"No Recent News for {company_name}",
            customer_name=customer_name,
            customer_metadata=customer_metadata
        )
    except Exception as e:
        logger.error(f"Error saving placeholder report: {str(e)}")
        return None

# ===== Report Management Functions =====

def check_existing_reports(savm_id: str) -> List[Dict[str, Any]]:
    """
    Check if reports already exist for a given SAVM ID.
    
    Args:
        savm_id: The SAVM ID to search for
        
    Returns:
        List[Dict[str, Any]]: List of report metadata dictionaries
    """
    try:
        # List all files in the GCS bucket
        all_files = list_files_in_gcs()
        
        # Filter for JSON files containing the SAVM ID
        report_files = []
        for file_path in all_files:
            if file_path.endswith('.json') and savm_id in file_path:
                # Extract timestamp from filename
                # Typically format: OUTCOMES_{customer}_{timestamp}.json
                timestamp_match = re.search(r'_(\d{8}_\d{6})\.json$', file_path)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        # Parse the timestamp
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        # Check if report is less than 30 days old
                        age_days = (datetime.now() - timestamp).days
                        
                        report_files.append({
                            'file_path': file_path,
                            'timestamp': timestamp,
                            'age_days': age_days,
                            'is_recent': age_days < 30
                        })
                    except:
                        # If timestamp parsing fails, include without details
                        report_files.append({
                            'file_path': file_path,
                            'timestamp': None,
                            'age_days': None,
                            'is_recent': False
                        })
        
        # Sort by timestamp (newest first)
        report_files.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
        
        return report_files
    
    except Exception as e:
        logger.error(f"Error checking existing reports for SAVM ID {savm_id}: {str(e)}")
        return []

def get_report_urls(base_path: str) -> Dict[str, str]:
    """
    Generate URLs for all report formats based on a base path.
    
    Args:
        base_path: The base path of the report (without extension)
        
    Returns:
        Dict[str, str]: Dictionary with URLs for different formats
    """
    try:
        result = {}
        
        # Check for related files with the same base name but different extensions
        extensions = ['.md', '.docx', '.json']
        
        for ext in extensions:
            file_path = f"{base_path}{ext}"
            
            # Generate a signed URL for this file
            url = generate_signed_url(file_path, expiration_days=7)
            
            if url:
                if ext == '.md':
                    result["markdown_url"] = url
                elif ext == '.docx':
                    result["docx_url"] = url
                elif ext == '.json':
                    result["json_url"] = url
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating report URLs: {str(e)}")
        return {}

def load_report_json(file_path: str) -> Dict[str, Any]:
    """
    Load a report from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict[str, Any]: The report data
    """
    try:
        # Check if this is a GCS path or local path
        if file_path.startswith(GCS_FOLDER) or not os.path.exists(file_path):
            # Assume GCS path
            from utils.storage import download_blob_as_string
            content = download_blob_as_string(file_path)
            if content:
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                return json.loads(content)
        else:
            # Local file
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading report from {file_path}: {str(e)}")
        return {
            "error": f"Failed to load report: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
