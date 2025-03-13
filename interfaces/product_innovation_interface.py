#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Product Innovation Interface

This module provides an interface for integrating the Product Innovation Agent
with the existing business aggregator. It allows users to run product innovation
analysis to identify underpriced products in the enterprise networking manufacturing space.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Import the ProductInnovationAgent
from agents.product_innovation_agent import ProductInnovationAgent, PRODUCT_CATEGORIES, TARGET_MANUFACTURERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("product_innovation_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProductInnovationInterface:
    """
    Interface for running product innovation analysis and integrating with the business aggregator.
    
    This class provides methods for:
    1. Running product innovation analysis for specific categories or manufacturers
    2. Generating reports on underpriced products
    3. Saving and loading analysis results
    """
    
    def __init__(self):
        """Initialize the Product Innovation Interface."""
        self.agent = ProductInnovationAgent()
        self.reports_dir = os.getenv("LOCAL_REPORTS_DIR", "reports")
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.join(self.reports_dir, "product_innovation"), exist_ok=True)
    
    async def run_analysis(self, category: Optional[str] = None, manufacturer: Optional[str] = None) -> Dict[str, Any]:
        """
        Run product innovation analysis to identify underpriced products.
        
        Args:
            category: Optional category to focus on (if None, all categories are analyzed)
            manufacturer: Optional manufacturer to focus on (if None, all manufacturers are analyzed)
            
        Returns:
            Dictionary with the analysis report
        """
        logger.info(f"Running product innovation analysis for category={category}, manufacturer={manufacturer}")
        
        try:
            # Run the analysis using the ProductInnovationAgent
            report = await self.agent.analyze_products(category, manufacturer)
            
            # Save the report
            self._save_report(report, category, manufacturer)
            
            return report
        except Exception as e:
            logger.error(f"Error running product innovation analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _save_report(self, report: Dict[str, Any], category: Optional[str] = None, manufacturer: Optional[str] = None) -> str:
        """
        Save the analysis report to a JSON file.
        
        Args:
            report: The analysis report to save
            category: Optional category used in the analysis
            manufacturer: Optional manufacturer used in the analysis
            
        Returns:
            Path to the saved report file
        """
        try:
            # Create a filename based on the category and manufacturer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_parts = ["product_innovation"]
            
            if category:
                # Clean the category name for use in filename
                clean_category = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in category)
                filename_parts.append(clean_category)
            
            if manufacturer:
                # Clean the manufacturer name for use in filename
                clean_manufacturer = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in manufacturer)
                filename_parts.append(clean_manufacturer)
            
            filename_parts.append(timestamp)
            filename = "_".join(filename_parts) + ".json"
            
            # Create the full file path
            file_path = os.path.join(self.reports_dir, "product_innovation", filename)
            
            # Save the report to the file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved product innovation report to {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving product innovation report: {str(e)}")
            return ""
    
    def get_available_categories(self) -> List[str]:
        """
        Get the list of available product categories for analysis.
        
        Returns:
            List of product categories
        """
        return PRODUCT_CATEGORIES
    
    def get_available_manufacturers(self) -> List[str]:
        """
        Get the list of available manufacturers for analysis.
        
        Returns:
            List of manufacturers
        """
        return TARGET_MANUFACTURERS
    
    def get_recent_reports(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a list of recent product innovation reports.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            List of report metadata dictionaries
        """
        try:
            # Get the path to the product innovation reports directory
            reports_dir = os.path.join(self.reports_dir, "product_innovation")
            
            # Check if the directory exists
            if not os.path.exists(reports_dir):
                return []
            
            # Get all JSON files in the directory
            report_files = [
                f for f in os.listdir(reports_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(reports_dir, f))
            ]
            
            # Sort by modification time (newest first)
            report_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)),
                reverse=True
            )
            
            # Get metadata for each report
            reports = []
            for filename in report_files[:limit]:
                file_path = os.path.join(reports_dir, filename)
                
                try:
                    # Extract timestamp from filename
                    timestamp_match = filename.split("_")[-1].split(".")[0]
                    if len(timestamp_match) == 15:  # Format: YYYYMMDD_HHMMSS
                        timestamp = datetime.strptime(timestamp_match, "%Y%m%d_%H%M%S")
                    else:
                        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Extract category and manufacturer from filename
                    parts = filename.split("_")
                    category = None
                    manufacturer = None
                    
                    if len(parts) > 3:  # product_innovation_category_manufacturer_timestamp.json
                        category = parts[2]
                        manufacturer = parts[3]
                    elif len(parts) > 2:  # product_innovation_category_timestamp.json
                        category = parts[2]
                    
                    # Load the report summary
                    with open(file_path, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                    
                    summary = report_data.get("summary", {})
                    
                    reports.append({
                        "filename": filename,
                        "file_path": file_path,
                        "timestamp": timestamp.isoformat(),
                        "category": category,
                        "manufacturer": manufacturer,
                        "total_products": summary.get("total_products_analyzed", 0),
                        "underpriced_count": summary.get("underpriced_products_count", 0),
                        "average_underpriced_percentage": summary.get("average_underpriced_percentage", 0)
                    })
                except Exception as e:
                    logger.error(f"Error processing report file {filename}: {str(e)}")
            
            return reports
        except Exception as e:
            logger.error(f"Error getting recent reports: {str(e)}")
            return []
    
    def load_report(self, file_path: str) -> Dict[str, Any]:
        """
        Load a product innovation report from a file.
        
        Args:
            file_path: Path to the report file
            
        Returns:
            The report data as a dictionary
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            
            return report_data
        except Exception as e:
            logger.error(f"Error loading report from {file_path}: {str(e)}")
            return {
                "error": f"Failed to load report: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

async def interactive_mode():
    """Run the product innovation analysis in interactive mode."""
    interface = ProductInnovationInterface()
    
    print("\n=== Product Innovation Analysis ===\n")
    print("This tool identifies underpriced products in the enterprise networking manufacturing space.")
    
    # Show available categories
    categories = interface.get_available_categories()
    print("\nAvailable Categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    # Get category selection
    print("\nSelect a category (enter number, or 0 for all categories):")
    category_input = input("> ")
    
    selected_category = None
    try:
        category_index = int(category_input)
        if 1 <= category_index <= len(categories):
            selected_category = categories[category_index - 1]
    except ValueError:
        # If input is not a number, check if it matches a category name
        if category_input in categories:
            selected_category = category_input
    
    # Show available manufacturers
    manufacturers = interface.get_available_manufacturers()
    print("\nAvailable Manufacturers:")
    for i, manufacturer in enumerate(manufacturers, 1):
        print(f"{i}. {manufacturer}")
    
    # Get manufacturer selection
    print("\nSelect a manufacturer (enter number, or 0 for all manufacturers):")
    manufacturer_input = input("> ")
    
    selected_manufacturer = None
    try:
        manufacturer_index = int(manufacturer_input)
        if 1 <= manufacturer_index <= len(manufacturers):
            selected_manufacturer = manufacturers[manufacturer_index - 1]
    except ValueError:
        # If input is not a number, check if it matches a manufacturer name
        if manufacturer_input in manufacturers:
            selected_manufacturer = manufacturer_input
    
    # Confirm analysis
    print("\nRunning product innovation analysis with the following parameters:")
    print(f"Category: {selected_category or 'All categories'}")
    print(f"Manufacturer: {selected_manufacturer or 'All manufacturers'}")
    print("\nThis may take several minutes. Please wait...\n")
    
    # Run the analysis
    report = await interface.run_analysis(selected_category, selected_manufacturer)
    
    # Display the results
    if "error" in report:
        print(f"Error: {report['error']}")
    else:
        summary = report.get("summary", {})
        print("\n=== Analysis Results ===\n")
        print(f"Total products analyzed: {summary.get('total_products_analyzed', 0)}")
        print(f"Underpriced products found: {summary.get('underpriced_products_count', 0)} ({summary.get('underpriced_products_percentage', 0):.2f}%)")
        print(f"Average underpriced percentage: {summary.get('average_underpriced_percentage', 0):.2f}%\n")
        
        print("Top Underpriced Products:")
        for i, product in enumerate(report.get("top_underpriced_products", [])[:5], 1):
            print(f"{i}. {product.get('manufacturer', 'Unknown')} {product.get('name', 'Unknown')} {product.get('model', 'Unknown')}")
            print(f"   Category: {product.get('category', 'Unknown')}")
            print(f"   Underpriced by: {product.get('underpriced_percentage', 0):.2f}%")
            print(f"   Assessment: {product.get('assessment', 'No assessment available')[:100]}...\n")
        
        print("Key Insights:")
        insights = report.get("insights", "No insights available")
        # Print the first 500 characters of insights with ellipsis
        print(f"{insights[:500]}...\n")
        
        # Get the report file path
        report_files = interface.get_recent_reports(limit=1)
        if report_files:
            print(f"Full report saved to: {report_files[0]['file_path']}")

async def main():
    """Main entry point for the product innovation interface."""
    try:
        # Check for command line arguments
        import sys
        
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            # Run in interactive mode
            await interactive_mode()
        else:
            # Get optional category and manufacturer from command line arguments
            category = sys.argv[1] if len(sys.argv) > 1 else None
            manufacturer = sys.argv[2] if len(sys.argv) > 2 else None
            
            # Initialize the interface
            interface = ProductInnovationInterface()
            
            # Run the analysis
            report = await interface.run_analysis(category, manufacturer)
            
            # Print the report summary
            print("\n=== Product Innovation Analysis Report ===\n")
            
            if "error" in report:
                print(f"Error: {report['error']}")
            else:
                summary = report.get("summary", {})
                print(f"Total products analyzed: {summary.get('total_products_analyzed', 0)}")
                print(f"Underpriced products found: {summary.get('underpriced_products_count', 0)} ({summary.get('underpriced_products_percentage', 0):.2f}%)")
                print(f"Average underpriced percentage: {summary.get('average_underpriced_percentage', 0):.2f}%\n")
                
                print("Top Underpriced Products:")
                for i, product in enumerate(report.get("top_underpriced_products", [])[:5], 1):
                    print(f"{i}. {product.get('manufacturer', 'Unknown')} {product.get('name', 'Unknown')} {product.get('model', 'Unknown')}")
                    print(f"   Category: {product.get('category', 'Unknown')}")
                    print(f"   Underpriced by: {product.get('underpriced_percentage', 0):.2f}%")
                    print(f"   Assessment: {product.get('assessment', 'No assessment available')[:100]}...\n")
                
                print("Key Insights:")
                insights = report.get("insights", "No insights available")
                # Print the first 500 characters of insights with ellipsis
                print(f"{insights[:500]}...\n")
                
                # Get the report file path
                report_files = interface.get_recent_reports(limit=1)
                if report_files:
                    print(f"Full report saved to: {report_files[0]['file_path']}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
