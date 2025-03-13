#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Script for Product Innovation Module

This script demonstrates how to use the Product Innovation module to identify
underpriced products in the enterprise networking manufacturing space.

Usage:
    python test_product_innovation.py [category] [manufacturer]

Examples:
    # Test with all categories and manufacturers (limited scope for testing)
    python test_product_innovation.py

    # Test with a specific category
    python test_product_innovation.py "network switches"

    # Test with a specific manufacturer
    python test_product_innovation.py "" "Cisco"

    # Test with a specific category and manufacturer
    python test_product_innovation.py "firewall systems" "Palo Alto Networks"
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ProductInnovationInterface
from product_innovation_interface import ProductInnovationInterface

async def run_test(category: Optional[str] = None, manufacturer: Optional[str] = None, test_mode: bool = True):
    """
    Run a test of the Product Innovation module.
    
    Args:
        category: Optional category to focus on
        manufacturer: Optional manufacturer to focus on
        test_mode: If True, limit the scope of the analysis for faster testing
    """
    print("\n=== Product Innovation Module Test ===\n")
    print(f"Testing with category: {category or 'All categories'}")
    print(f"Testing with manufacturer: {manufacturer or 'All manufacturers'}")
    
    if test_mode:
        print("\nRunning in TEST MODE - analysis scope is limited for faster results")
    
    print("\nInitializing Product Innovation Interface...")
    interface = ProductInnovationInterface()
    
    # In test mode, we'll modify the agent to use a smaller subset of categories and manufacturers
    if test_mode and not category and not manufacturer:
        # Override the agent's categories and manufacturers for testing
        # This makes the test run much faster by limiting the scope
        print("Limiting test scope to one category and one manufacturer...")
        category = "network switches"
        manufacturer = "Cisco"
    
    print(f"\nRunning product innovation analysis for {category or 'all categories'} and {manufacturer or 'all manufacturers'}...")
    print("This may take several minutes. Please wait...\n")
    
    start_time = datetime.now()
    
    try:
        # Run the analysis
        report = await interface.run_analysis(category, manufacturer)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nAnalysis completed in {duration:.2f} seconds")
        
        # Check if the analysis was successful
        if "error" in report:
            print(f"\nError: {report['error']}")
            return False
        
        # Display the results
        summary = report.get("summary", {})
        print("\n=== Analysis Results ===\n")
        print(f"Total products analyzed: {summary.get('total_products_analyzed', 0)}")
        print(f"Underpriced products found: {summary.get('underpriced_products_count', 0)} ({summary.get('underpriced_products_percentage', 0):.2f}%)")
        print(f"Average underpriced percentage: {summary.get('average_underpriced_percentage', 0):.2f}%\n")
        
        # Display top underpriced products
        top_products = report.get("top_underpriced_products", [])
        if top_products:
            print("Top Underpriced Products:")
            for i, product in enumerate(top_products[:3], 1):
                print(f"{i}. {product.get('manufacturer', 'Unknown')} {product.get('name', 'Unknown')} {product.get('model', 'Unknown')}")
                print(f"   Category: {product.get('category', 'Unknown')}")
                print(f"   Underpriced by: {product.get('underpriced_percentage', 0):.2f}%")
                print(f"   Assessment: {product.get('assessment', 'No assessment available')[:100]}...\n")
        else:
            print("No underpriced products found.\n")
        
        # Display category summaries
        categories = report.get("categories", {})
        if categories:
            print("Category Summaries:")
            for category_name, category_data in categories.items():
                print(f"- {category_name}: {category_data.get('count', 0)} products, avg {category_data.get('average_underpriced_percentage', 0):.2f}% underpriced")
            print()
        
        # Display manufacturer summaries
        manufacturers = report.get("manufacturers", {})
        if manufacturers:
            print("Manufacturer Summaries:")
            for manufacturer_name, manufacturer_data in manufacturers.items():
                print(f"- {manufacturer_name}: {manufacturer_data.get('count', 0)} products, avg {manufacturer_data.get('average_underpriced_percentage', 0):.2f}% underpriced")
            print()
        
        # Display a snippet of the insights
        insights = report.get("insights", "No insights available")
        print("Insights Preview:")
        print(f"{insights[:300]}...\n")
        
        # Get the report file path
        report_files = interface.get_recent_reports(limit=1)
        if report_files:
            print(f"Full report saved to: {report_files[0]['file_path']}")
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nTest failed after {duration:.2f} seconds")
        print(f"Error: {str(e)}")
        return False

async def main():
    """Main entry point for the test script."""
    try:
        # Get optional category and manufacturer from command line arguments
        category = sys.argv[1] if len(sys.argv) > 1 else None
        manufacturer = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Run the test
        success = await run_test(category, manufacturer)
        
        # Exit with appropriate status code
        sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
