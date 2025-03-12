#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Company URL Finder Tool

This module provides functionality to find and verify company URLs, match them with
customer records in the database, and generate research reports about the companies.

Key Features:
- URL discovery using Tavily search API
- Human-in-the-loop verification of URLs and company matches
- Integration with Google Cloud Storage for Excel database and report storage
- Report generation using the research_agent module
"""

import os
import re
import json
import asyncio
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from urllib.parse import urlparse
from google.cloud import storage
from google.oauth2 import service_account

# Load environment variables from .env file before any other imports
load_dotenv()

# Set Azure OpenAI environment variables with the correct names
# This resolves the environment variable name mismatch issue
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))  # Use Azure key for OpenAI
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "https://phx-sales-ai.openai.azure.com/")
os.environ["AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Import for Tavily search
from tavily import TavilyClient

# Import for LangGraph (human-in-the-loop)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import necessary functions from research_agent.py
from research_agent import (
    process_url, 
    load_customer_data, 
    get_gcs_client, 
    extract_company_name
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("company_url_finder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API keys are available
if not TAVILY_API_KEY:
    logger.error("TAVILY_API_KEY not found in environment variables.")
    raise ValueError("Missing TAVILY_API_KEY in environment variables")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Set up Google Cloud Storage credentials path
CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets", "google-credentials-dev.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# Extract bucket information from env vars
GCS_BUCKET_PATH = os.getenv("OUTCOMES_PATH", "gs://sales-ai-dev-outcomes-6f1ce1c")
GCS_BUCKET_NAME = GCS_BUCKET_PATH.replace("gs://", "").split("/")[0]
GCS_FOLDER = "news-reports"
GCS_EXCEL_FOLDER = "data"  # Folder where Excel files are stored in GCS

# Always use GCS for Excel files
USE_GCS_EXCEL = True

# Flag to control whether to print links to files
PRINT_FILE_LINKS = True

async def find_company_url(company_name: str) -> str:
    """
    Use Tavily to find the official website URL for a company.
    
    This function leverages the Tavily search API to find the most likely
    official website for a given company. It performs intelligent extraction
    and filtering of URLs from search results.
    
    Search strategy:
    1. First attempt: search for "what is the official website URL for {company_name}"
    2. Extract URLs from the answer and context
    3. If no URLs found, second attempt: search for "{company_name} official website"
    4. Filter and score URLs based on relevance to the company name
    
    Args:
        company_name (str): The name of the company to search for.
        
    Returns:
        str: The official website URL of the company, or None if not found.
    """
    logger.info(f"Finding URL for company: {company_name}")
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Create a search query for the company's official website
    query = f"what is the official website URL for {company_name}"
    
    try:
        # STEP 1: Perform the initial search
        logger.info(f"Performing Tavily search with query: '{query}'")
        search_result = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            include_domains=None,
            exclude_domains=None
        )
        
        # STEP 2: Extract potential URLs from search results
        urls = []
        
        # Extract from the answer if available
        if "answer" in search_result and search_result["answer"]:
            # Use regex to find URLs in the answer
            url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
            found_urls = re.findall(url_pattern, search_result["answer"])
            urls.extend(found_urls)
            
            # Look for domain patterns like "example.com" that might not have http/https
            domain_pattern = r'(?<!\S)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?!\S)'
            domain_urls = re.findall(domain_pattern, search_result["answer"])
            urls.extend([f"https://{domain}" for domain in domain_urls])
        
        # Also check the context sections
        if "context" in search_result and search_result["context"]:
            for context_item in search_result["context"]:
                if "content" in context_item and context_item["content"]:
                    # Use regex to find URLs in the content
                    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
                    found_urls = re.findall(url_pattern, context_item["content"])
                    urls.extend(found_urls)
                    
                    # Look for domain patterns like "example.com" that might not have http/https
                    domain_pattern = r'(?<!\S)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?!\S)'
                    domain_urls = re.findall(domain_pattern, context_item["content"])
                    urls.extend([f"https://{domain}" for domain in domain_urls if domain not in company_name.lower()])
                
                # Also check URLs directly in the context
                if "url" in context_item and context_item["url"]:
                    urls.append(context_item["url"])
        
        # STEP 3: Try a second search if no URLs found
        if not urls:
            logger.info("No URLs found in initial search, trying direct website search")
            # Try a more direct query
            direct_query = f"{company_name} official website"
            search_result = tavily_client.search(
                query=direct_query,
                search_depth="advanced",
                include_answer=True
            )
            
            # Check the URLs in the search results
            if "results" in search_result:
                for result in search_result["results"]:
                    if "url" in result:
                        urls.append(result["url"])
        
        # STEP 4: Filter and prioritize URLs
        # Remove duplicates
        urls = list(set(urls))
        
        # Filter URLs to keep only those likely to be company websites
        company_name_parts = company_name.lower().split()
        filtered_urls = []
        
        for url in urls:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if not domain:
                # Handle cases where the URL might just be a domain without http/https
                domain = parsed_url.path.lower()
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check if any part of the company name is in the domain
            domain_without_tld = domain.split('.')[0] if '.' in domain else domain
            
            # Check if domain contains company name or company name contains domain
            if any(part in domain_without_tld for part in company_name_parts) or domain_without_tld in company_name.lower():
                filtered_urls.append(url)
        
        # STEP 5: Score and select the best URL
        # If we have filtered URLs, use those, otherwise fall back to all URLs
        potential_urls = filtered_urls if filtered_urls else urls
        
        # If we still have multiple URLs, prioritize by domain relevance
        if len(potential_urls) > 1:
            # Score each URL based on domain similarity to company name
            url_scores = []
            for url in potential_urls:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                if not domain:
                    domain = parsed_url.path.lower()
                
                # Remove www. prefix if present
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Score based on domain parts matching company name parts
                score = 0
                domain_parts = re.split(r'[.-]', domain)
                for part in domain_parts:
                    if part and any(company_part in part or part in company_part for company_part in company_name_parts):
                        score += 1
                
                # Prefer .com domains
                if domain.endswith('.com'):
                    score += 0.5
                
                # Prefer shorter domains
                score -= 0.1 * len(domain)
                
                url_scores.append((url, score))
            
            # Sort by score, descending
            url_scores.sort(key=lambda x: x[1], reverse=True)
            potential_urls = [url for url, _ in url_scores]
        
        # Return the best URL found, or None if no URLs were found
        if potential_urls:
            logger.info(f"Found potential URLs: {potential_urls}")
            logger.info(f"Selected best URL: {potential_urls[0]}")
            return potential_urls[0]
        else:
            logger.warning(f"No URLs found for {company_name}")
            return None
    
    except Exception as e:
        logger.error(f"Error finding URL for {company_name}: {str(e)}")
        return None

def extract_domain(url: str) -> str:
    """
    Extract and normalize the domain from a URL.
    
    This function handles various URL formats, including URLs with or without protocols,
    and normalizes the domain by:
    1. Adding https:// if missing
    2. Parsing the domain from the URL
    3. Converting to lowercase
    4. Removing www. prefix if present
    
    The function includes fallback logic for cases where standard URL parsing fails.
    
    Args:
        url (str): The URL to process.
        
    Returns:
        str: The normalized domain, or empty string if input is invalid or parsing fails.
    """
    if not url:
        return ""
    
    # Add https:// if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    try:
        # Use standard URL parsing
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except:
        # Fallback method if standard parsing fails
        logger.warning(f"Standard URL parsing failed for {url}, using fallback method")
        try:
            # Simple approach: remove protocols and split by first /
            url = url.lower()
            url = url.replace('https://', '').replace('http://', '')
            if url.startswith('www.'):
                url = url[4:]
            return url.split('/')[0]
        except Exception as e:
            logger.error(f"Domain extraction failed completely for {url}: {str(e)}")
            return ""

def find_best_url_match(target_url: str, urls: List[str]) -> str:
    """
    Find the best match for a URL in a list of URLs using domain comparison.
    
    This function attempts to find the most similar domain among a list of URLs
    that matches the target URL. It extracts domains, normalizes them, and compares
    them using both exact matches and similarity-based scoring.
    
    Matching strategy:
    1. First attempt exact domain matches
    2. If no exact matches, try flexible domain matching with similarity scoring
    3. Higher scoring is given to shorter domain name differences and longer common prefixes
    
    Args:
        target_url (str): The URL to match.
        urls (List[str]): The list of URLs to search in.
        
    Returns:
        str: The best matching URL from the list, or None if no match found.
    """
    if not target_url or not urls:
        return None
    
    # Extract and normalize the target domain
    target_domain = extract_domain(target_url)
    if not target_domain:
        return None
    
    # Create patterns for matching
    # Pattern 1: Exact domain match
    exact_pattern = rf'^{re.escape(target_domain)}$'
    
    # Pattern 2: Domain with optional www prefix and any protocol
    flexible_pattern = rf'^(?:www\.)?{re.escape(target_domain)}(?:/.*)?$'
    
    # First, try exact matches (highest priority)
    for url in urls:
        url_domain = extract_domain(url)
        if url_domain == target_domain:
            return url
    
    # Second, try regex pattern matches with similarity scoring
    matches = []
    for url in urls:
        url_domain = extract_domain(url)
        
        # Skip empty domains
        if not url_domain:
            continue
        
        # Check if domains match regardless of www prefix
        if re.match(flexible_pattern, url_domain) or re.match(flexible_pattern, target_domain):
            # Calculate similarity score (lower is better)
            similarity = 0
            
            # Penalize length differences
            similarity += abs(len(url_domain) - len(target_domain))
            
            # Check common prefix length - longer common prefix gets better score
            common_prefix_len = 0
            for i in range(min(len(url_domain), len(target_domain))):
                if url_domain[i] == target_domain[i]:
                    common_prefix_len += 1
                else:
                    break
            
            # Higher common prefix is better (lower score)
            similarity -= common_prefix_len
            
            matches.append((url, similarity))
    
    # Sort by similarity score (lower is better)
    if matches:
        matches.sort(key=lambda x: x[1])
        return matches[0][0]
    
    # If no matches, return None
    return None

async def verify_url_human_in_loop(company_name: str, url: str) -> str:
    """
    Verify the company URL using human input.
    
    This function implements the human-in-the-loop verification process for ensuring
    the correct URL is associated with a company. If the URL is incorrect, the user
    can provide the correct URL.
    
    Flow:
    1. Present the discovered URL to the user and ask for verification
    2. If verified, return the original URL
    3. If not verified, prompt for the correct URL
    4. Attempt to extract/clean the provided URL
    5. Return the corrected URL or a constructed fallback URL
    
    Args:
        company_name (str): The name of the company.
        url (str): The discovered URL to verify.
        
    Returns:
        str: The verified URL, potentially modified by the human.
    """
    if not url:
        return None
    
    # Ask the user to verify the URL
    logger.info(f"Requesting verification for URL: {url} for company: {company_name}")
    print(f"\nI found the website for {company_name} to be: {url}")
    print("\nIs this correct? (yes/no)")
    response = input("> ").lower()
    
    # If the user confirms the URL is correct, return it
    if "yes" in response:
        logger.info(f"User verified URL: {url}")
        return url
    else:
        # If the URL is incorrect, ask for the correct one
        logger.info("User indicated URL is incorrect, requesting correction")
        print("Please provide the correct URL:")
        corrected_url = input("> ")
        
        # Process the user input to ensure it's a valid URL
        if not corrected_url.startswith("http") and "." not in corrected_url:
            # Try to find a URL pattern in their response
            url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
            found_urls = re.findall(url_pattern, corrected_url)
            
            # Also look for domain patterns without http/https
            domain_pattern = r'(?<!\S)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?!\S)'
            domain_urls = re.findall(domain_pattern, corrected_url)
            
            # Use found URL or construct a simple one if none found
            if found_urls:
                corrected_url = found_urls[0]
                logger.info(f"Extracted URL from input: {corrected_url}")
            elif domain_urls:
                corrected_url = f"https://{domain_urls[0]}"
                logger.info(f"Extracted domain and created URL: {corrected_url}")
            else:
                logger.warning("No valid URL found in user input, creating simple URL from company name")
                # Create a simplified URL from the company name as a fallback
                company_url = company_name.lower().replace(" ", "")
                corrected_url = f"https://{company_url}.com"
                logger.info(f"Created simplified URL: {corrected_url}")
        
        # Ensure the URL has a protocol (https://)
        if corrected_url and not corrected_url.startswith("http"):
            corrected_url = f"https://{corrected_url}"
            logger.info(f"Added https:// prefix to URL: {corrected_url}")
        
        return corrected_url

def download_from_gcs(gcs_path, local_path):
    """
    Download a file from Google Cloud Storage to a local path.
    
    This function handles the process of downloading a file from GCS to a local path,
    accounting for different path formats and properly extracting the blob path.
    
    Args:
        gcs_path (str): The path to the file in GCS. Can be a full path like 
                       "gs://bucket-name/path/to/file" or just "path/to/file".
        local_path (str): The local path where the file should be saved.
        
    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        # Initialize GCS client
        client = get_gcs_client()
        if not client:
            logger.error("Failed to initialize GCS client")
            return False
            
        # Extract the blob path from the full GCS path
        if gcs_path.startswith("gs://"):
            # Remove the gs:// prefix and bucket name
            parts = gcs_path.replace("gs://", "").split("/", 1)
            if len(parts) < 2:
                logger.error(f"Invalid GCS path format: {gcs_path}")
                return False
            blob_path = parts[1]
        else:
            # Assume the path is already a blob path
            blob_path = gcs_path
        
        # Get the bucket and blob
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_path)
        
        # Download the file
        logger.info(f"Downloading file from gs://{GCS_BUCKET_NAME}/{blob_path} to {local_path}")
        blob.download_to_filename(local_path)
        
        # Verify the file was downloaded
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            logger.info(f"File successfully downloaded to {local_path} ({file_size} bytes)")
            return True
        else:
            logger.error(f"File download did not create the expected local file: {local_path}")
            return False
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        return False

def search_database_for_url(url: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Search the database for a URL and return the matching SAVM ID with metadata.
    
    This function attempts to find a company in the Excel database that matches 
    the given URL. It downloads the Excel file from GCS (if configured) and performs
    matching using domain comparison.
    
    Flow:
    1. Determine the Excel file path (GCS or local)
    2. Load customer data from Excel
    3. Find the best URL match using domain comparison
    4. Extract SAVM_ID and company metadata from the matching row
    5. Return the SAVM_ID and metadata
    
    Args:
        url (str): The URL to search for.
        
    Returns:
        Tuple[Optional[str], Optional[Dict]]: A tuple containing the SAVM ID and customer metadata,
                                           or (None, None) if not found.
    """
    try:
        # Define the Excel file path
        excel_filename = "Customer Parquet top 80 select hierarchy for test.xlsx"
        
        # Step 1: Determine Excel file location (GCS or local)
        if USE_GCS_EXCEL:
            # If using GCS, download the Excel file from GCS
            logger.info("Using Excel file from Google Cloud Storage")
            
            # Create a temporary directory for the downloaded file
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="company_url_finder_")
            local_excel_path = os.path.join(temp_dir, excel_filename)
            
            # GCS path to the Excel file
            gcs_excel_path = f"{GCS_EXCEL_FOLDER}/{excel_filename}"
            
            # Download the file
            download_success = download_from_gcs(gcs_excel_path, local_excel_path)
            
            if not download_success:
                logger.error(f"Failed to download Excel file from GCS. Falling back to local file.")
                # Fall back to the local file
                excel_path = excel_filename
            else:
                excel_path = local_excel_path
        else:
            # Use the local Excel file
            excel_path = excel_filename
        
        # Step 2: Load customer data from Excel
        logger.info(f"Loading customer data from: {excel_path}")
        df = pd.read_excel(excel_path)
        
        logger.info(f"Loaded {len(df)} rows from Excel file")
        
        # Filter for valid websites (not blank or ".")
        df = df[df['WEBSITE'].notna()]  # Remove NaN values
        df = df[df['WEBSITE'] != "."]   # Remove "." values
        df = df[df['WEBSITE'] != ""]    # Remove empty strings
        
        # Step 3: Find the best URL match
        # Extract all URLs from the database
        urls = df['WEBSITE'].tolist()
        best_match = find_best_url_match(url, urls)
        
        # Step 4: Extract data from the matching row (if found)
        if best_match:
            # Get the row with the matching URL
            row = df[df['WEBSITE'] == best_match].iloc[0]
            
            # Extract the SAVM ID from SAVM_NAME_WITH_ID if available
            savm_id = None
            customer_name = None
            if 'SAVM_NAME_WITH_ID' in row and pd.notna(row['SAVM_NAME_WITH_ID']):
                savm_name_with_id = row['SAVM_NAME_WITH_ID']
                # SAVM_NAME_WITH_ID format is typically "Company Name (SAVMID)"
                match = re.search(r'\(([^)]+)\)', savm_name_with_id)
                if match:
                    savm_id = match.group(1)
                customer_name = savm_name_with_id
            
            # Step 5: Create metadata dictionary
            metadata = {}
            for column in df.columns:
                if pd.notna(row[column]):
                    if isinstance(row[column], (int, float, str, bool)):
                        metadata[column] = row[column]
                    else:
                        metadata[column] = str(row[column])
            
            logger.info(f"Found match in database: {best_match} -> SAVM ID: {savm_id}")
            return savm_id, metadata
        else:
            logger.warning(f"No matching URL found in database for {url}")
            return None, None
    
    except Exception as e:
        logger.error(f"Error searching database for URL {url}: {str(e)}")
        return None, None

def list_files_in_gcs(prefix: str = None) -> List[str]:
    """
    List files in the Google Cloud Storage bucket with an optional prefix.
    
    This function retrieves a list of all files in the configured GCS bucket,
    optionally filtered by a prefix. The prefix is appended to the GCS_FOLDER
    to form the full path prefix.
    
    Args:
        prefix (str, optional): The prefix to filter files within the GCS_FOLDER.
                               Defaults to None.
        
    Returns:
        List[str]: A list of file paths (blob names) in the bucket.
    """
    try:
        # Initialize GCS client
        client = get_gcs_client()
        if not client:
            logger.error("Failed to initialize GCS client")
            return []
        
        # Access the bucket
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Set the full prefix including folder
        full_prefix = f"{GCS_FOLDER}/" if not prefix else f"{GCS_FOLDER}/{prefix}"
        logger.info(f"Listing files in gs://{GCS_BUCKET_NAME}/{full_prefix}")
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=full_prefix)
        
        # Extract the blob names
        file_paths = [blob.name for blob in blobs]
        
        logger.info(f"Found {len(file_paths)} files in GCS bucket")
        return file_paths
    
    except Exception as e:
        logger.error(f"Error listing files in GCS: {str(e)}")
        return []

def check_existing_reports(savm_id: str) -> List[Dict]:
    """
    Check if reports already exist for a given SAVM ID in Google Cloud Storage.
    
    This function looks for existing report files in GCS that contain the specified
    SAVM ID. It extracts timestamp information from the filenames to determine 
    report age and sorts reports by recency.
    
    Args:
        savm_id (str): The SAVM ID to search for.
        
    Returns:
        List[Dict]: A list of report metadata dictionaries containing:
            - file_path: The GCS path to the file
            - timestamp: The report generation time (if available)
            - age_days: The age of the report in days
            - is_recent: Whether the report is less than 30 days old
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

def download_report_from_gcs(file_path: str) -> Dict:
    """
    Download a report from Google Cloud Storage and parse its JSON content.
    
    This function connects to GCS, downloads the specified report file as a string,
    and parses it as JSON.
    
    Args:
        file_path (str): The path to the file in GCS.
        
    Returns:
        Dict: The report content as a dictionary, or None if download fails.
    """
    try:
        # Initialize GCS client
        client = get_gcs_client()
        if not client:
            logger.error("Failed to initialize GCS client")
            return None
        
        # Get the bucket and blob
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        # Download as string
        logger.info(f"Downloading report from GCS: gs://{GCS_BUCKET_NAME}/{file_path}")
        json_str = blob.download_as_string()
        
        # Parse JSON content
        report_content = json.loads(json_str)
        logger.info(f"Successfully downloaded and parsed report ({len(json_str)} bytes)")
        
        return report_content
    
    except Exception as e:
        logger.error(f"Error downloading report from GCS {file_path}: {str(e)}")
        return None

async def verify_savm_id_match(verified_url: str, savm_id: str, savm_name: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Verify the SAVM_ID match with human input.
    
    This function implements the human-in-the-loop verification process for ensuring
    the correct company (SAVM_ID) is matched to a URL. If the initial match is incorrect,
    it allows the user to search for the correct company by name.
    
    Flow:
    1. Show the user the matched company and ask for verification
    2. If verified, return the SAVM_ID and metadata
    3. If not verified, prompt for correct company name
    4. Search for matches in the database using the provided name
    5. If multiple matches found, let the user select the correct one
    6. Return the selected company's SAVM_ID and metadata, or None if no match
    
    Args:
        verified_url (str): The verified URL
        savm_id (str): The initially matched SAVM ID
        savm_name (str): The SAVM name with ID
        df (pd.DataFrame): The dataframe containing customer data
        
    Returns:
        Tuple[Optional[str], Optional[Dict]]: A tuple containing the verified SAVM ID and customer metadata,
                                           or (None, None) if no match is found.
    """
    # Log the verification request
    logger.info(f"Requesting verification for SAVM_ID match: {savm_name} for URL: {verified_url}")
    
    # Ask the user to verify the match
    print(f"\nI found that this URL: {verified_url} matches with: {savm_name}")
    print("Is this the correct company? (yes/no)")
    response = input("> ").lower()
    
    # If the user verifies the match, return the original SAVM_ID and metadata
    if "yes" in response:
        logger.info(f"User verified SAVM_ID match: {savm_id}")
        
        # Get the row with the matching SAVM ID
        row = df[df['SAVM_NAME_WITH_ID'].str.contains(savm_id, regex=False, na=False)].iloc[0]
        
        # Create a metadata dictionary with all available fields
        metadata = {}
        for column in df.columns:
            if pd.notna(row[column]):
                if isinstance(row[column], (int, float, str, bool)):
                    metadata[column] = row[column]
                else:
                    metadata[column] = str(row[column])
                    
        return savm_id, metadata
    else:
        # If the user indicates the match is incorrect, search by name instead
        logger.info("User indicated SAVM_ID match is incorrect, searching by name instead")
        print("\nPlease enter the correct company name to search for:")
        company_name = input("> ")
        
        # Search for similar company names in the SAVM_NAME_WITH_ID column
        name_matches = []
        
        # Convert input to lowercase for case-insensitive matching
        company_name_lower = company_name.lower()
        
        # First try exact substring match
        mask = df['SAVM_NAME_WITH_ID'].fillna("").str.lower().str.contains(company_name_lower, regex=False)
        name_matches = df[mask]
        
        # If no matches, try fuzzy regex matching
        if len(name_matches) == 0:
            # Split the company name into tokens and create a regex pattern
            tokens = company_name_lower.split()
            if tokens:
                # Create a pattern that looks for all tokens in any order
                pattern = '.*'.join(f"({re.escape(token)})" for token in tokens)
                mask = df['SAVM_NAME_WITH_ID'].fillna("").str.lower().str.contains(pattern, regex=True)
                name_matches = df[mask]
        
        # Process the matches found (if any)
        if len(name_matches) > 0:
            # If multiple matches, let the user choose
            if len(name_matches) > 1:
                print("\nMultiple matches found. Please select the correct one:")
                for i, row in enumerate(name_matches.itertuples()):
                    print(f"{i+1}. {row.SAVM_NAME_WITH_ID}")
                
                try:
                    selection = int(input("> ")) - 1
                    if 0 <= selection < len(name_matches):
                        selected_row = name_matches.iloc[selection]
                        
                        # Extract the SAVM ID from the selected row
                        match = re.search(r'\(([^)]+)\)', selected_row['SAVM_NAME_WITH_ID'])
                        if match:
                            savm_id = match.group(1)
                            
                            # Create metadata dictionary
                            metadata = {}
                            for column in df.columns:
                                if pd.notna(selected_row[column]):
                                    if isinstance(selected_row[column], (int, float, str, bool)):
                                        metadata[column] = selected_row[column]
                                    else:
                                        metadata[column] = str(selected_row[column])
                                        
                            logger.info(f"User selected SAVM_ID: {savm_id}")
                            return savm_id, metadata
                    else:
                        logger.warning("Invalid selection.")
                except (ValueError, IndexError):
                    logger.warning("Invalid input for selection.")
            else:
                # Single match found
                selected_row = name_matches.iloc[0]
                
                # Extract the SAVM ID
                match = re.search(r'\(([^)]+)\)', selected_row['SAVM_NAME_WITH_ID'])
                if match:
                    savm_id = match.group(1)
                    
                    # Create metadata dictionary
                    metadata = {}
                    for column in df.columns:
                        if pd.notna(selected_row[column]):
                            if isinstance(selected_row[column], (int, float, str, bool)):
                                metadata[column] = selected_row[column]
                            else:
                                metadata[column] = str(selected_row[column])
                                
                    logger.info(f"Found match by name: {selected_row['SAVM_NAME_WITH_ID']} -> SAVM ID: {savm_id}")
                    return savm_id, metadata
        
        # If no matches found or selection failed
        logger.warning(f"No matching company name found for '{company_name}'")
        return None, None

async def main():
    """Main entry point for the company URL finder."""
    try:
        # Get the company name from user input
        print("Enter the company name to search for: ")
        company_name = input("> ")
        logger.info(f"Starting search for company: {company_name}")
        
        # Find the company URL using Tavily
        url = await find_company_url(company_name)
        
        if not url:
            logger.error(f"Could not find URL for {company_name}. Exiting.")
            return
        
        # Verify the URL with human input
        verified_url = await verify_url_human_in_loop(company_name, url)
        
        if not verified_url:
            logger.error("URL verification failed. Exiting.")
            return
        
        logger.info(f"Verified URL for {company_name}: {verified_url}")
        
        # Search for the URL in the database
        savm_id, customer_metadata = search_database_for_url(verified_url)
        
        if not savm_id:
            logger.warning(f"No SAVM ID found for URL {verified_url}. Will still attempt to generate a report.")
            # Process the URL without SAVM ID
            customer_name = company_name
        else:
            # We have a SAVM ID, but verify it's the correct one
            logger.info(f"Found SAVM ID: {savm_id}")
            customer_name = customer_metadata.get('SAVM_NAME_WITH_ID', company_name)
            
            # Define the Excel file path
            excel_filename = "Customer Parquet top 80 select hierarchy for test.xlsx"
            
            if USE_GCS_EXCEL:
                # If using GCS, download the Excel file from GCS
                logger.info("Using Excel file from Google Cloud Storage")
                
                # Create a temporary directory for the downloaded file
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="company_url_finder_")
                local_excel_path = os.path.join(temp_dir, excel_filename)
                
                # GCS path to the Excel file
                gcs_excel_path = f"{GCS_EXCEL_FOLDER}/{excel_filename}"
                
                # Download the file
                download_success = download_from_gcs(gcs_excel_path, local_excel_path)
                
                if not download_success:
                    logger.error(f"Failed to download Excel file from GCS. Falling back to local file.")
                    # Fall back to the local file
                    excel_path = excel_filename
                else:
                    excel_path = local_excel_path
            else:
                # Use the local Excel file
                excel_path = excel_filename
            
            # Load the customer data from Excel
            logger.info(f"Loading customer data for verification: {excel_path}")
            df = pd.read_excel(excel_path)
            
            # Verify the SAVM ID match with human input
            verified_savm_id, verified_metadata = await verify_savm_id_match(
                verified_url, 
                savm_id, 
                customer_name, 
                df
            )
            
            if verified_savm_id:
                savm_id = verified_savm_id
                customer_metadata = verified_metadata
                customer_name = verified_metadata.get('SAVM_NAME_WITH_ID', company_name)
                logger.info(f"Using verified SAVM ID: {savm_id} for {customer_name}")
            else:
                logger.warning("SAVM ID verification failed. Will use the company name without metadata.")
                savm_id = None
                customer_metadata = None
                customer_name = company_name
            
            # If we have a verified SAVM ID, check for existing reports
            if savm_id:
                # Check for existing reports
                existing_reports = check_existing_reports(savm_id)
                
                if existing_reports and any(report['is_recent'] for report in existing_reports):
                    # We have a recent report (less than 30 days old)
                    recent_report = next(report for report in existing_reports if report['is_recent'])
                    logger.info(f"Found recent report ({recent_report['age_days']} days old): {recent_report['file_path']}")
                    
                    # Download the report
                    report_content = download_report_from_gcs(recent_report['file_path'])
                    
                    if report_content:
                        logger.info(f"Successfully downloaded existing report")
                        
                        # Display the report summary
                        logger.info(f"Company: {customer_name}")
                        logger.info(f"URL: {verified_url}")
                        logger.info(f"Report Age: {recent_report['age_days']} days")
                        logger.info(f"Report Path: gs://{GCS_BUCKET_NAME}/{recent_report['file_path']}")
                        
                        if 'content' in report_content:
                            # Extract the first 300 characters of content as a preview
                            content_preview = report_content['content'][:300] + "..." if len(report_content['content']) > 300 else report_content['content']
                            logger.info(f"Content Preview:\n{content_preview}")
                        
                        # Ask if user wants to generate a new report anyway
                        print("\nDo you want to generate a new report anyway? (yes/no): ")
                        regenerate = input("> ").lower() == 'yes'
                        
                        if not regenerate:
                            logger.info("User chose to use existing report")
                            
                            # Display all available report formats
                            print("\n=== Available Report Links ===")
                            
                            # Get all files in the bucket
                            all_files = list_files_in_gcs()
                            
                            # Check for related files with the same base name but different extensions
                            base_path = recent_report['file_path'].rsplit('.', 1)[0]
                            related_files = [f for f in all_files if f.startswith(base_path)]
                            
                            for file in related_files:
                                file_type = "Unknown"
                                if file.endswith('.md'):
                                    file_type = "Markdown"
                                elif file.endswith('.docx'):
                                    file_type = "Word Document"
                                elif file.endswith('.json'):
                                    file_type = "JSON Data"
                                    
                                # Generate a signed URL for this file
                                try:
                                    client = get_gcs_client()
                                    bucket = client.bucket(GCS_BUCKET_NAME)
                                    blob = bucket.blob(file)
                                    signed_url = blob.generate_signed_url(
                                        version="v4",
                                        expiration=timedelta(days=7),
                                        method="GET"
                                    )
                                    print(f"{file_type}: {signed_url}")
                                except Exception as e:
                                    logger.error(f"Error generating signed URL for {file}: {str(e)}")
                                    print(f"{file_type}: gs://{GCS_BUCKET_NAME}/{file} (Error generating signed URL)")
                            
                            return
                    
                    # If we got here, either the report download failed or user wants to regenerate
                    logger.info("Generating a new report...")
        
        # Generate a new report using process_url
        logger.info(f"Generating new report for {customer_name} ({verified_url})")
        
        # Create a company-specific topic
        company_name_short = extract_company_name(verified_url, customer_name)
        topic = f"""Research and analyze news specifically about {company_name_short}.

The report structure should follow this exact order:

1. Company News (REQUIRES RESEARCH)
   - Any substantial news about the specific company {company_name_short}
   - Recent acquisitions, mergers, or partnerships involving {company_name_short}
   - Leadership changes or restructuring at {company_name_short}

2. News about their IT Priorities (REQUIRES RESEARCH)
   - {company_name_short}'s specific IT challenges, pain points, and desired outcomes
   - {company_name_short}'s technology investments or initiatives
   - {company_name_short}'s digital transformation efforts and IT strategy
   - Specific IT projects, implementations, or challenges faced by {company_name_short}

3. Discovery Questions for Cisco Sellers
   - Based on the findings in sections 1 and 2, create 3-5 thoughtful discovery questions that Cisco sellers can use to start conversations with {company_name_short}
   - Questions should relate to their IT challenges, pain points, or initiatives identified in the research
   - Questions should be open-ended and designed to uncover opportunities for Cisco solutions

Note: Focus ONLY on {company_name_short}. Do NOT include general industry trends or news about other companies unless they directly relate to {company_name_short}.
"""
        
        # Process the URL to generate a report
        try:
            result = await process_url(
                url=verified_url, 
                topic=topic, 
                customer_name=customer_name, 
                customer_metadata=customer_metadata,
                planner_model="gpt-4o",
                planner_provider="azure_openai"
            )
            
            if result:
                logger.info(f"Successfully generated new report")
                
                # Display the results
                print("\n=== New Research Report Generated ===")
                
                if 'markdown_gcs_url' in result:
                    print(f"Markdown Report: {result['markdown_gcs_url']}")
                
                if 'docx_gcs_url' in result:
                    print(f"Word Document: {result['docx_gcs_url']}")
                
                if 'json_gcs_url' in result:
                    print(f"JSON Report: {result['json_gcs_url']}")
            else:
                logger.error("Failed to generate report")
                logger.error("Check the logs for details.")
        
        except Exception as e:
            logger.error(f"Error processing URL {verified_url}: {str(e)}")
            logger.error(f"Error generating report: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 