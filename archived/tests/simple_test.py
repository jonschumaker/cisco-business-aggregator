#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script for the Product Innovation Agent.
"""

import asyncio
from product_innovation_agent import ProductInnovationAgent

async def main():
    """Test the Product Innovation Agent."""
    print("Initializing ProductInnovationAgent...")
    agent = ProductInnovationAgent()
    print("ProductInnovationAgent initialized successfully!")
    
    # Test a simple analysis with a very limited scope
    print("\nRunning a simple analysis with limited scope...")
    report = await agent.analyze_products(
        category="network switches",
        manufacturer="Cisco"
    )
    
    if "error" in report:
        print(f"Error: {report['error']}")
    else:
        print("Analysis completed successfully!")
        print(f"Report: {report}")

if __name__ == "__main__":
    asyncio.run(main())
