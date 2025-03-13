#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models Package

This package provides data models for the application.
It includes models for products, reports, companies, and other data structures.

Usage:
    from models.product import Product
    from models.report import Report
    from models.company import Company
"""

# Import key classes for easy access
from models.product import Product, ProductAnalysis, PriceValueAssessment
from models.report import Report, ReportSection, ReportMetadata
from models.company import Company, CompanyMetadata
