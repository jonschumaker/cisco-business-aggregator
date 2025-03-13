#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to check if the reports directory exists and create a test file in it.
"""

import os
import datetime

# Check if reports directory exists
reports_dir = "reports"
if os.path.exists(reports_dir):
    print(f"Reports directory exists: {os.path.abspath(reports_dir)}")
else:
    print(f"Reports directory does not exist. Creating it...")
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Created reports directory: {os.path.abspath(reports_dir)}")

# Create a test file in the reports directory
test_file = os.path.join(reports_dir, "test_file.txt")
with open(test_file, "w") as f:
    f.write(f"Test file created at {datetime.datetime.now()}")

print(f"Created test file: {os.path.abspath(test_file)}")

# List all files in the reports directory
print("\nFiles in reports directory:")
for root, dirs, files in os.walk(reports_dir):
    for file in files:
        print(f"  {os.path.join(root, file)}")
