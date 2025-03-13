#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Directories

This script creates the necessary directories for the Cisco Business Aggregator project.
"""

import os
import sys

def create_directories():
    """Create the necessary directories for the project."""
    # Define the directories to create
    directories = [
        "reports",
        "cache",
        "logs",
        "secrets"
    ]
    
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create each directory
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Error creating directory {dir_path}: {str(e)}")
        else:
            print(f"Directory already exists: {dir_path}")
    
    # Create a .gitkeep file in each directory to ensure it's tracked by Git
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        gitkeep_path = os.path.join(dir_path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            try:
                with open(gitkeep_path, "w") as f:
                    f.write("# This file ensures the directory is tracked by Git\n")
                print(f"Created .gitkeep file in: {dir_path}")
            except Exception as e:
                print(f"Error creating .gitkeep file in {dir_path}: {str(e)}")
    
    print("\nDirectory setup complete!")
    print("\nNext steps:")
    print("1. Create a .env file with your API keys")
    print("2. Install dependencies with: pip install -r requirements.txt")
    print("3. Run the application with: python main.py")

if __name__ == "__main__":
    create_directories()
