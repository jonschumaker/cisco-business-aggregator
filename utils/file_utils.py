#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Utilities

This module provides utility functions for file operations.
It handles operations like file conversion, report creation, and file formatting.

Key features:
- Markdown to Word conversion
- Markdown to JSON conversion
- Report creation and formatting
- Source standardization
"""

import os
import re
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

def markdown_to_word(markdown_content: str, output_path: str, title: str = "Report") -> str:
    """
    Convert markdown content to a Word document.
    
    Args:
        markdown_content: Markdown content to convert
        output_path: Path to save the Word document
        title: Title of the document
        
    Returns:
        str: Path to the created Word document
    """
    try:
        import pypandoc
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert markdown to docx
        pypandoc.convert_text(
            markdown_content,
            'docx',
            format='md',
            outputfile=output_path,
            extra_args=[f'--metadata=title:{title}']
        )
        
        logger.info(f"Converted markdown to Word document: {output_path}")
        return output_path
    except ImportError:
        logger.error("pypandoc not installed. Install with: pip install pypandoc")
        raise ImportError("pypandoc not installed")
    except Exception as e:
        logger.error(f"Error converting markdown to Word: {str(e)}")
        raise

def markdown_to_json(markdown_content: str, url: str, topic: str, 
                    customer_name: Optional[str] = None, 
                    company_name: Optional[str] = None,
                    customer_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert markdown content to a structured JSON format.
    
    Args:
        markdown_content: Markdown content to convert
        url: URL associated with the report
        topic: Topic of the report
        customer_name: Optional customer name
        company_name: Optional company name
        customer_metadata: Optional dictionary of customer metadata
        
    Returns:
        Dict[str, Any]: Structured JSON representation of the markdown content
    """
    try:
        # Extract sections from markdown
        sections = []
        current_section = None
        current_content = []
        current_level = 0
        current_title = ""
        
        # Process each line
        for line in markdown_content.split('\n'):
            # Check if this is a heading
            heading_match = re.match(r'^(#+)\s+(.+)$', line)
            if heading_match:
                # If we have a current section, save it
                if current_section is not None:
                    sections.append({
                        "level": current_level,
                        "title": current_title,
                        "content": '\n'.join(current_content).strip(),
                        "section_type": determine_section_type(current_title)
                    })
                
                # Start a new section
                current_level = len(heading_match.group(1))
                current_title = heading_match.group(2).strip()
                current_content = []
                current_section = current_title
            else:
                # Add line to current section
                if current_section is not None:
                    current_content.append(line)
        
        # Add the last section if there is one
        if current_section is not None:
            sections.append({
                "level": current_level,
                "title": current_title,
                "content": '\n'.join(current_content).strip(),
                "section_type": determine_section_type(current_title)
            })
        
        # Extract sources from sections
        for section in sections:
            section["sources"] = extract_sources(section["content"])
        
        # Create metadata
        metadata = {
            "url": url,
            "topic": topic,
            "generation_date": datetime.now().isoformat()
        }
        
        # Add optional fields if they exist
        if customer_name:
            metadata["customer_name"] = customer_name
        
        if company_name:
            metadata["company_name"] = company_name
        
        # Add customer metadata if provided
        if customer_metadata:
            for key, value in customer_metadata.items():
                metadata[key] = value
        
        # Create the final JSON structure
        result = {
            "id": generate_uuid(),
            "metadata": metadata,
            "sections": sections
        }
        
        return result
    except Exception as e:
        logger.error(f"Error converting markdown to JSON: {str(e)}")
        return {
            "id": generate_uuid(),
            "metadata": {
                "url": url,
                "topic": topic,
                "generation_date": datetime.now().isoformat(),
                "error": str(e)
            },
            "sections": []
        }

def determine_section_type(title: str) -> str:
    """
    Determine the type of a section based on its title.
    
    Args:
        title: Title of the section
        
    Returns:
        str: Type of the section
    """
    title_lower = title.lower()
    
    if "introduction" in title_lower or "overview" in title_lower:
        return "introduction"
    elif "conclusion" in title_lower or "summary" in title_lower:
        return "conclusion"
    elif "news" in title_lower or "announcement" in title_lower or "press release" in title_lower:
        return "company_news"
    elif "product" in title_lower and ("launch" in title_lower or "release" in title_lower or "new" in title_lower):
        return "product_launch"
    elif "financial" in title_lower or "earnings" in title_lower or "revenue" in title_lower:
        return "financial"
    elif "partnership" in title_lower or "collaboration" in title_lower or "alliance" in title_lower:
        return "partnership"
    elif "acquisition" in title_lower or "merger" in title_lower:
        return "acquisition"
    elif "leadership" in title_lower or "executive" in title_lower or "management" in title_lower:
        return "leadership"
    elif "strategy" in title_lower or "vision" in title_lower or "mission" in title_lower:
        return "strategy"
    elif "innovation" in title_lower or "research" in title_lower or "development" in title_lower:
        return "innovation"
    elif "market" in title_lower or "industry" in title_lower or "trend" in title_lower:
        return "market"
    elif "challenge" in title_lower or "issue" in title_lower or "problem" in title_lower:
        return "challenge"
    elif "opportunity" in title_lower or "potential" in title_lower:
        return "opportunity"
    elif "source" in title_lower or "reference" in title_lower:
        return "sources"
    elif "url" in title_lower:
        return "url"
    else:
        return "other"

def extract_sources(content: str) -> List[str]:
    """
    Extract sources from content.
    
    Args:
        content: Content to extract sources from
        
    Returns:
        List[str]: List of sources
    """
    sources = []
    
    # Look for sources in the format [1]: http://example.com
    source_matches = re.findall(r'\[(\d+)\]:\s*(.+)', content)
    for _, source in source_matches:
        sources.append(source.strip())
    
    # Look for URLs in the content
    url_matches = re.findall(r'https?://[^\s\)]+', content)
    for url in url_matches:
        if url not in sources:
            sources.append(url)
    
    return sources

def standardize_sources_in_markdown(markdown_content: str) -> str:
    """
    Standardize sources in markdown content.
    
    Args:
        markdown_content: Markdown content to standardize
        
    Returns:
        str: Standardized markdown content
    """
    # Extract all URLs from the content
    urls = re.findall(r'https?://[^\s\)]+', markdown_content)
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    # Replace URLs in the content with numbered references
    content = markdown_content
    for i, url in enumerate(unique_urls):
        content = content.replace(url, f"[{i+1}]")
    
    # Add sources section at the end
    content += "\n\n## Sources\n\n"
    for i, url in enumerate(unique_urls):
        content += f"[{i+1}]: {url}\n"
    
    return content

def create_markdown_report(url: str, content: str, topic: str, 
                          customer_name: Optional[str] = None, 
                          customer_metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a markdown report with proper formatting.
    
    Args:
        url: URL associated with the report
        content: Main content of the report
        topic: Topic of the report
        customer_name: Optional customer name
        customer_metadata: Optional dictionary of customer metadata
        
    Returns:
        str: Formatted markdown report
    """
    # Extract company name from URL or customer name
    company_name = extract_company_name(url, customer_name)
    
    # Create the report title
    if customer_name:
        title = f"# Research Report for {customer_name}"
    else:
        title = f"# Research Report on {company_name}"
    
    # Add generation date
    date_str = datetime.now().strftime("%B %d, %Y")
    date_line = f"*Generated on {date_str}*"
    
    # Add URL
    url_line = f"## URL: {url}"
    
    # Add topic
    topic_line = f"## Topic: {topic}"
    
    # Add customer metadata if provided
    metadata_lines = []
    if customer_metadata:
        metadata_lines.append("## Customer Information")
        for key, value in customer_metadata.items():
            if key != "SAVM_NAME_WITH_ID" and value:  # Skip SAVM_NAME_WITH_ID as it's already in the title
                metadata_lines.append(f"- **{key}**: {value}")
    
    # Combine all parts
    report_parts = [
        title,
        date_line,
        url_line,
        topic_line
    ]
    
    if metadata_lines:
        report_parts.extend(metadata_lines)
    
    report_parts.append("## Content")
    report_parts.append(content)
    
    # Join all parts with double newlines
    report = "\n\n".join(report_parts)
    
    # Standardize sources
    report = standardize_sources_in_markdown(report)
    
    return report

def create_error_report(url: str, error_message: str, company_name: str, 
                       customer_name: Optional[str] = None) -> str:
    """
    Create an error report when research fails.
    
    Args:
        url: URL that was being researched
        error_message: Error message
        company_name: Name of the company
        customer_name: Optional customer name
        
    Returns:
        str: Error report content
    """
    # Create the report title
    if customer_name:
        title = f"# Error Report for {customer_name}"
    else:
        title = f"# Error Report for {company_name}"
    
    # Add generation date
    date_str = datetime.now().strftime("%B %d, %Y")
    date_line = f"*Generated on {date_str}*"
    
    # Add URL
    url_line = f"## URL: {url}"
    
    # Add error message
    error_section = f"## Error\n\n{error_message}"
    
    # Add recommendation
    recommendation = """
## Recommendation

Please try again later or contact support if the issue persists.
"""
    
    # Combine all parts
    report_parts = [
        title,
        date_line,
        url_line,
        error_section,
        recommendation
    ]
    
    # Join all parts with double newlines
    report = "\n\n".join(report_parts)
    
    return report

def create_placeholder_report(url: str, lookback_days: int, company_name: str, 
                             customer_name: Optional[str] = None) -> str:
    """
    Create a placeholder report when no significant news is found.
    
    Args:
        url: URL that was being researched
        lookback_days: Number of days that were searched
        company_name: Name of the company
        customer_name: Optional customer name
        
    Returns:
        str: Placeholder report content
    """
    # Create the report title
    if customer_name:
        title = f"# No Recent News for {customer_name}"
    else:
        title = f"# No Recent News for {company_name}"
    
    # Add generation date
    date_str = datetime.now().strftime("%B %d, %Y")
    date_line = f"*Generated on {date_str}*"
    
    # Add URL
    url_line = f"## URL: {url}"
    
    # Add no news message
    no_news_message = f"""
## No Significant News Found

No significant news or announcements were found for {company_name} in the past {lookback_days} days.

This could be because:
- The company has not made any major announcements recently
- The company's recent activities are not widely reported
- The search parameters may need to be adjusted

Please check back later for updates or modify your search criteria.
"""
    
    # Combine all parts
    report_parts = [
        title,
        date_line,
        url_line,
        no_news_message
    ]
    
    # Join all parts with double newlines
    report = "\n\n".join(report_parts)
    
    return report

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

def generate_uuid() -> str:
    """
    Generate a UUID.
    
    Returns:
        str: Generated UUID
    """
    import uuid
    return str(uuid.uuid4())
