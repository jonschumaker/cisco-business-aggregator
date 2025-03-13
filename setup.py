#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements
with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="cisco-business-aggregator",
    version="0.1.0",
    description="A modular Python application for aggregating business intelligence, researching companies, and analyzing product innovation opportunities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cisco Team",
    author_email="example@cisco.com",
    url="https://github.com/cisco/business-aggregator",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cisco-business-aggregator=main:main",
        ],
    },
)
