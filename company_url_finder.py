#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# Flag to control whether to print links to files
PRINT_FILE_LINKS = True

async def find_company_url(company_name: str) -> str:
    """
    Use Tavily to find the official website URL for a company.
    
    Args:
        company_name (str): The name of the company to search for.
        
    Returns:
        str: The official website URL of the company.
    """
    logger.info(f"Finding URL for company: {company_name}")
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Create a search query for the company's official website
    query = f"what is the official website URL for {company_name}"
    
    try:
        # Perform the search
        search_result = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            include_domains=None,
            exclude_domains=None
        )
        
        # Extract potential URLs from the answer and context
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
        
        # If we found no URLs in the answer or context, try direct Tavily searches
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
        
        # Remove duplicates and filter irrelevant URLs
        urls = list(set(urls))
        
        # Filter URLs to keep only those likely to be company websites
        # 1. Check if company name is in the domain (converting to lowercase for comparison)
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
        
        if potential_urls:
            logger.info(f"Found potential URLs: {potential_urls}")
            return potential_urls[0]
        else:
            logger.warning(f"No URLs found for {company_name}")
            return None
    
    except Exception as e:
        logger.error(f"Error finding URL for {company_name}: {str(e)}")
        return None

def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL and normalize it.
    
    Args:
        url (str): The URL to process.
        
    Returns:
        str: The normalized domain.
    """
    if not url:
        return ""
    
    # Add https:// if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except:
        # If parsing fails, try a simple approach
        url = url.lower()
        url = url.replace('https://', '').replace('http://', '')
        if url.startswith('www.'):
            url = url[4:]
        return url.split('/')[0]

def find_best_url_match(target_url: str, urls: List[str]) -> str:
    """
    Find the best match for a URL in a list of URLs using regex pattern matching.
    
    Args:
        target_url (str): The URL to match.
        urls (List[str]): The list of URLs to search in.
        
    Returns:
        str: The best matching URL from the list.
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
    
    # First, try exact matches
    for url in urls:
        url_domain = extract_domain(url)
        if url_domain == target_domain:
            return url
    
    # Second, try regex pattern matches
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
            
            # Check common prefix length
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
    Verify the company URL using human input (simplified without LangGraph).
    
    Args:
        company_name (str): The name of the company.
        url (str): The URL to verify.
        
    Returns:
        str: The verified URL, potentially modified by the human.
    """
    if not url:
        return None
    
    logger.info(f"Requesting verification for URL: {url} for company: {company_name}")
    print(f"\nI found the website for {company_name} to be: {url}")
    print("\nIs this correct? (yes/no)")
    response = input("> ").lower()
    
    if "yes" in response:
        logger.info(f"User verified URL: {url}")
        return url
    else:
        logger.info("User indicated URL is incorrect, requesting correction")
        print("Please provide the correct URL:")
        corrected_url = input("> ")
        
        # Try to extract a URL from the human's response if they didn't provide a clear URL
        if not corrected_url.startswith("http") and "." not in corrected_url:
            # Try to find a URL in their response
            url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
            found_urls = re.findall(url_pattern, corrected_url)
            
            # Also look for domain patterns without http/https
            domain_pattern = r'(?<!\S)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?!\S)'
            domain_urls = re.findall(domain_pattern, corrected_url)
            
            if found_urls:
                corrected_url = found_urls[0]
                logger.info(f"Extracted URL from input: {corrected_url}")
            elif domain_urls:
                corrected_url = f"https://{domain_urls[0]}"
                logger.info(f"Extracted domain and created URL: {corrected_url}")
            else:
                logger.warning("No valid URL found in user input, creating simple URL from company name")
                # Create a simplified URL from the company name
                company_url = company_name.lower().replace(" ", "")
                corrected_url = f"https://{company_url}.com"
                logger.info(f"Created simplified URL: {corrected_url}")
        
        # Add https:// if missing
        if corrected_url and not corrected_url.startswith("http"):
            corrected_url = f"https://{corrected_url}"
            logger.info(f"Added https:// prefix to URL: {corrected_url}")
        
        return corrected_url

def search_database_for_url(url: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Search the database for a URL and return the matching SAVM ID.
    
    Args:
        url (str): The URL to search for.
        
    Returns:
        Tuple[Optional[str], Optional[Dict]]: A tuple containing the SAVM ID and customer metadata,
                                           or (None, None) if not found.
    """
    try:
        # Load the customer data from Excel
        excel_path = "Customer Parquet top 80 select hierarchy for test.xlsx"
        df = pd.read_excel(excel_path)
        
        logger.info(f"Loaded {len(df)} rows from Excel file")
        
        # Filter for valid websites (not blank or ".")
        df = df[df['WEBSITE'].notna()]  # Remove NaN values
        df = df[df['WEBSITE'] != "."]   # Remove "." values
        df = df[df['WEBSITE'] != ""]    # Remove empty strings
        
        # Extract all URLs from the database
        urls = df['WEBSITE'].tolist()
        
        # Find the best match for the target URL
        best_match = find_best_url_match(url, urls)
        
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
            
            # Create a metadata dictionary with all available fields
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
    
    Args:
        prefix (str, optional): The prefix to filter files. Defaults to None.
        
    Returns:
        List[str]: A list of file paths.
    """
    try:
        client = get_gcs_client()
        if not client:
            logger.error("Failed to initialize GCS client")
            return []
        
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Set the full prefix including folder
        full_prefix = f"{GCS_FOLDER}/" if not prefix else f"{GCS_FOLDER}/{prefix}"
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=full_prefix)
        
        # Extract the blob names
        file_paths = [blob.name for blob in blobs]
        
        return file_paths
    
    except Exception as e:
        logger.error(f"Error listing files in GCS: {str(e)}")
        return []

def check_existing_reports(savm_id: str) -> List[Dict]:
    """
    Check if reports already exist for a given SAVM ID.
    
    Args:
        savm_id (str): The SAVM ID to search for.
        
    Returns:
        List[Dict]: A list of reports with their metadata.
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
    Download a report from Google Cloud Storage.
    
    Args:
        file_path (str): The path to the file in GCS.
        
    Returns:
        Dict: The report content as a dictionary.
    """
    try:
        client = get_gcs_client()
        if not client:
            logger.error("Failed to initialize GCS client")
            return None
        
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        # Download as string
        json_str = blob.download_as_string()
        
        # Parse JSON content
        report_content = json.loads(json_str)
        
        return report_content
    
    except Exception as e:
        logger.error(f"Error downloading report from GCS {file_path}: {str(e)}")
        return None

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
            # We have a SAVM ID, check for existing reports
            logger.info(f"Found SAVM ID: {savm_id}")
            customer_name = customer_metadata.get('SAVM_NAME_WITH_ID', company_name)
            
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
                customer_metadata=customer_metadata
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