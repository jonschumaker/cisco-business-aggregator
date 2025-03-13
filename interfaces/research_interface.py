#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Research Interface

This module provides a user interface for the research functionality.
It allows users to research companies and generate reports.

Key features:
- Command-line interface for research
- Interactive mode for guided research
- Report generation and viewing
"""

import os
import sys
import logging
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Union

# Import local modules
from config import settings, logging_config
from agents.research_agent import ResearchAgent
from agents.company_url_finder import CompanyURLFinder
from models.company import Company, CompanyMetadata

# Configure logging
logger = logging_config.get_logger(__name__)

class ResearchInterface:
    """
    Interface for the research functionality.
    
    This class provides methods for researching companies and generating reports.
    """
    
    def __init__(self):
        """
        Initialize the research interface.
        """
        self.research_agent = ResearchAgent()
        self.url_finder = CompanyURLFinder()
    
    async def run_research(self, 
                          url: Optional[str] = None, 
                          company_name: Optional[str] = None,
                          customer_name: Optional[str] = None,
                          savm_id: Optional[str] = None,
                          lookback_days: int = 30,
                          interactive: bool = False) -> Dict[str, str]:
        """
        Run research on a company.
        
        Args:
            url: URL of the company website
            company_name: Name of the company
            customer_name: Name of the customer
            savm_id: SAVM ID
            lookback_days: Number of days to look back for news
            interactive: Whether to run in interactive mode
            
        Returns:
            Dict[str, str]: Dictionary with report file paths/URLs
        """
        logger.info(f"Running research with URL: {url}, company name: {company_name}")
        
        # Create customer metadata
        customer_metadata = {}
        if customer_name:
            customer_metadata["SAVM_NAME_WITH_ID"] = customer_name
        if savm_id:
            customer_metadata["SAVM_ID"] = savm_id
        
        # If URL is not provided but company name is, find the URL
        if not url and company_name:
            logger.info(f"Finding URL for company: {company_name}")
            url = await self.url_finder.find_company_url(company_name)
            
            if not url:
                logger.error(f"Could not find URL for company: {company_name}")
                return {"error": f"Could not find URL for company: {company_name}"}
            
            logger.info(f"Found URL for company {company_name}: {url}")
        
        # If neither URL nor company name is provided and not in interactive mode, return error
        if not url and not company_name and not interactive:
            logger.error("URL or company name is required")
            return {"error": "URL or company name is required"}
        
        # If in interactive mode and neither URL nor company name is provided, prompt for input
        if interactive and not url and not company_name:
            company_name = input("Enter company name: ")
            
            if not company_name:
                logger.error("Company name is required")
                return {"error": "Company name is required"}
            
            logger.info(f"Finding URL for company: {company_name}")
            url = await self.url_finder.find_company_url(company_name)
            
            if not url:
                logger.error(f"Could not find URL for company: {company_name}")
                return {"error": f"Could not find URL for company: {company_name}"}
            
            logger.info(f"Found URL for company {company_name}: {url}")
        
        # Run the research
        result = await self.research_agent.research_url(
            url=url,
            lookback_days=lookback_days,
            customer_name=customer_name,
            customer_metadata=customer_metadata
        )
        
        return result
    
    async def check_existing_reports(self, savm_id: str) -> List[Dict[str, Any]]:
        """
        Check for existing reports for a SAVM ID.
        
        Args:
            savm_id: SAVM ID to check
            
        Returns:
            List[Dict[str, Any]]: List of report metadata dictionaries
        """
        from utils.report_utils import check_existing_reports
        
        logger.info(f"Checking for existing reports for SAVM ID: {savm_id}")
        
        # Check for existing reports
        reports = check_existing_reports(savm_id)
        
        return reports

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parsing.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Research Interface")
    
    # Research command
    parser.add_argument("--url", help="URL to research")
    parser.add_argument("--company", help="Company name to research")
    parser.add_argument("--customer-name", help="Customer name")
    parser.add_argument("--savm-id", help="SAVM ID")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    return parser

async def main() -> None:
    """
    Main entry point.
    """
    # Set up argument parsing
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create the research interface
    interface = ResearchInterface()
    
    # Run the research
    result = await interface.run_research(
        url=args.url,
        company_name=args.company,
        customer_name=args.customer_name,
        savm_id=args.savm_id,
        lookback_days=args.days,
        interactive=args.interactive
    )
    
    # Print the result
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Research completed successfully. Report saved to:")
        for key, value in result.items():
            if key.endswith("_path") or key.endswith("_url"):
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
