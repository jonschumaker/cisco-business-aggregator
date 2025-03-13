#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point

This module serves as the main entry point for the application.
It provides command-line interfaces for running the various agents and tools.

Usage:
    python main.py research --url https://example.com
    python main.py find-url --company "Example Company"
    python main.py product-innovation --manufacturer "Cisco"
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
from agents.product_innovation_agent import ProductInnovationAgent
from interfaces.product_innovation_interface import ProductInnovationInterface
from interfaces.research_interface import ResearchInterface
from models.company import Company, CompanyMetadata
from utils.cache import save_cache_to_disk

# Configure logging
logger = logging_config.get_logger(__name__)

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parsing.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Cisco Business Aggregator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Run the research agent")
    research_parser.add_argument("--url", required=True, help="URL to research")
    research_parser.add_argument("--customer-name", help="Customer name")
    research_parser.add_argument("--savm-id", help="SAVM ID")
    research_parser.add_argument("--days", type=int, default=30, help="Number of days to look back")
    
    # Find URL command
    find_url_parser = subparsers.add_parser("find-url", help="Run the company URL finder")
    find_url_parser.add_argument("--company", required=True, help="Company name")
    find_url_parser.add_argument("--customer-name", help="Customer name")
    find_url_parser.add_argument("--savm-id", help="SAVM ID")
    
    # Product innovation command
    product_parser = subparsers.add_parser("product-innovation", help="Run the product innovation agent")
    product_parser.add_argument("--manufacturer", required=True, help="Manufacturer name")
    product_parser.add_argument("--category", help="Product category")
    product_parser.add_argument("--model", help="Product model")
    
    # Check reports command
    check_reports_parser = subparsers.add_parser("check-reports", help="Check for existing reports")
    check_reports_parser.add_argument("--savm-id", required=True, help="SAVM ID to check")
    
    # Clear cache command
    subparsers.add_parser("clear-cache", help="Clear the cache")
    
    return parser

async def run_research(args: argparse.Namespace) -> None:
    """
    Run the research agent.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Running research agent for URL: {args.url}")
    
    # Create customer metadata
    customer_metadata = {}
    if args.customer_name:
        customer_metadata["SAVM_NAME_WITH_ID"] = args.customer_name
    if args.savm_id:
        customer_metadata["SAVM_ID"] = args.savm_id
    
    # Create the research agent
    agent = ResearchAgent()
    
    # Run the research
    result = await agent.research_url(
        url=args.url,
        lookback_days=args.days,
        customer_name=args.customer_name,
        customer_metadata=customer_metadata
    )
    
    # Print the result
    if result:
        print(f"Research completed successfully. Report saved to:")
        for key, value in result.items():
            if key.endswith("_path") or key.endswith("_url"):
                print(f"  {key}: {value}")
    else:
        print("Research failed. Check the logs for details.")

async def run_find_url(args: argparse.Namespace) -> None:
    """
    Run the company URL finder.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Running company URL finder for company: {args.company}")
    
    # Create the company URL finder
    finder = CompanyURLFinder()
    
    # Find the URL
    url = await finder.find_company_url(args.company)
    
    # Print the result
    if url:
        print(f"Company URL found: {url}")
        
        # If customer name or SAVM ID is provided, run research
        if args.customer_name or args.savm_id:
            print(f"Running research on the found URL...")
            
            # Create customer metadata
            customer_metadata = {}
            if args.customer_name:
                customer_metadata["SAVM_NAME_WITH_ID"] = args.customer_name
            if args.savm_id:
                customer_metadata["SAVM_ID"] = args.savm_id
            
            # Create the research agent
            agent = ResearchAgent()
            
            # Run the research
            result = await agent.research_url(
                url=url,
                lookback_days=30,
                customer_name=args.customer_name,
                customer_metadata=customer_metadata
            )
            
            # Print the result
            if result:
                print(f"Research completed successfully. Report saved to:")
                for key, value in result.items():
                    if key.endswith("_path") or key.endswith("_url"):
                        print(f"  {key}: {value}")
            else:
                print("Research failed. Check the logs for details.")
    else:
        print(f"Could not find URL for company: {args.company}")

async def run_product_innovation(args: argparse.Namespace) -> None:
    """
    Run the product innovation agent.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Running product innovation agent for manufacturer: {args.manufacturer}")
    
    # Create the product innovation agent
    agent = ProductInnovationAgent()
    
    # Run the product innovation analysis
    try:
        result = await agent.analyze_products(
            manufacturer=args.manufacturer,
            category=args.category,
            model=args.model
        )
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"summary": str(result)}
    except Exception as e:
        logger.error(f"Error in product innovation analysis: {str(e)}")
        result = {"error": str(e)}
    
    # Print the result
    if result:
        print(f"Product innovation analysis completed successfully.")
        
        # Handle different result types
        if isinstance(result, dict):
            if "error" in result:
                print(f"Error: {result['error']}")
            elif "summary" in result and isinstance(result["summary"], str):
                # Handle the case where summary is a string
                print(result["summary"])
            else:
                # Handle the case where result is a dictionary with expected structure
                summary = result.get("summary", {})
                if isinstance(summary, dict):
                    print(f"Total products analyzed: {summary.get('total_products_analyzed', 0)}")
                    print(f"Underpriced products found: {summary.get('underpriced_products_count', 0)} ({summary.get('underpriced_products_percentage', 0):.2f}%)")
                    print(f"Average underpriced percentage: {summary.get('average_underpriced_percentage', 0):.2f}%\n")
                else:
                    print(f"Summary: {summary}")
                
                # Print top underpriced products if available
                if "top_underpriced_products" in result and result["top_underpriced_products"]:
                    print("Top Underpriced Products:")
                    for i, product in enumerate(result.get("top_underpriced_products", [])[:5], 1):
                        print(f"{i}. {product.get('manufacturer', 'Unknown')} {product.get('name', 'Unknown')} {product.get('model', 'Unknown')}")
                        print(f"   Category: {product.get('category', 'Unknown')}")
                        print(f"   Underpriced by: {product.get('underpriced_percentage', 0):.2f}%")
                        print(f"   Assessment: {product.get('assessment', 'No assessment available')[:100]}...\n")
        elif isinstance(result, str):
            # Handle the case where result is a string
            print(result)
        else:
            # Handle unexpected result types
            print(f"Unexpected result type: {type(result)}")
            print(str(result))
    else:
        print("Product innovation analysis failed. Check the logs for details.")

async def check_reports(args: argparse.Namespace) -> None:
    """
    Check for existing reports.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Checking for existing reports for SAVM ID: {args.savm_id}")
    
    # Import the report utilities
    from utils.report_utils import check_existing_reports, get_report_urls
    
    # Check for existing reports
    reports = check_existing_reports(args.savm_id)
    
    # Print the result
    if reports:
        print(f"Found {len(reports)} reports for SAVM ID: {args.savm_id}")
        for i, report in enumerate(reports):
            print(f"Report {i+1}:")
            print(f"  File: {report['file_path']}")
            if report['timestamp']:
                print(f"  Date: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Age: {report['age_days']} days")
                print(f"  Recent: {'Yes' if report['is_recent'] else 'No'}")
            
            # Get URLs for the report
            base_path = report['file_path'].rsplit('.', 1)[0]
            urls = get_report_urls(base_path)
            if urls:
                print("  URLs:")
                for key, url in urls.items():
                    print(f"    {key}: {url}")
            
            print()
    else:
        print(f"No reports found for SAVM ID: {args.savm_id}")

def clear_cache() -> None:
    """
    Clear the cache.
    """
    logger.info("Clearing cache")
    
    # Import the cache utilities
    from utils.cache import clear_cache, save_cache_to_disk
    
    # Clear the cache
    clear_cache()
    
    # Save the empty cache to disk
    save_cache_to_disk()
    
    print("Cache cleared successfully.")

async def main() -> None:
    """
    Main entry point.
    """
    # Set up argument parsing
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    try:
        if args.command == "research":
            await run_research(args)
        elif args.command == "find-url":
            await run_find_url(args)
        elif args.command == "product-innovation":
            await run_product_innovation(args)
        elif args.command == "check-reports":
            await check_reports(args)
        elif args.command == "clear-cache":
            clear_cache()
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        # Save the cache to disk
        save_cache_to_disk()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
