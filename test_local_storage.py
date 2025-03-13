#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for local storage functionality in research_agent.py
"""

import os
import asyncio
from dotenv import load_dotenv
from research_agent import save_markdown_report

# Load environment variables
load_dotenv()

# Ensure USE_LOCAL_STORAGE is set to true
os.environ["USE_LOCAL_STORAGE"] = "true"
os.environ["LOCAL_REPORTS_DIR"] = "reports"

async def test_local_storage():
    """Test the local storage functionality by saving a simple report."""
    print("Testing local storage functionality...")
    print(f"USE_LOCAL_STORAGE: {os.getenv('USE_LOCAL_STORAGE')}")
    print(f"LOCAL_REPORTS_DIR: {os.getenv('LOCAL_REPORTS_DIR')}")
    
    # Create a simple test report
    url = "https://example.com"
    content = """# Test Report

This is a test report to verify that local storage is working correctly.

## Section 1

This is section 1 of the test report.

## Section 2

This is section 2 of the test report.
"""
    topic = "Test Topic"
    customer_name = "Test Company"
    
    # Save the report
    result = save_markdown_report(
        url=url,
        content=content,
        topic=topic,
        customer_name=customer_name
    )
    
    # Print the result
    if result:
        print("\nReport saved successfully!")
        print("File paths:")
        for key, path in result.items():
            print(f"  {key}: {path}")
            
        # Verify the files exist
        for key, path in result.items():
            if os.path.exists(path):
                print(f"  ✓ File exists: {path}")
            else:
                print(f"  ✗ File does not exist: {path}")
    else:
        print("Failed to save report.")

if __name__ == "__main__":
    asyncio.run(test_local_storage())
