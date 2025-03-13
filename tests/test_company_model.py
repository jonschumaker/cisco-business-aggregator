#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Company model.

This module contains tests for the Company and CompanyMetadata models in models/company.py.
"""

import pytest
from datetime import datetime

# Import the models to test
from models.company import Company, CompanyMetadata

# Test CompanyMetadata
def test_company_metadata():
    """Test the CompanyMetadata class."""
    # Create a CompanyMetadata instance
    metadata = CompanyMetadata(
        savm_id="12345",
        savm_name_with_id="Example Company (12345)",
        sales_level_1="Level 1",
        sales_level_2="Level 2",
        sales_level_3="Level 3",
        sales_level_4="Level 4",
        additional_metadata={"custom_field": "custom_value"}
    )
    
    # Test that the attributes are set correctly
    assert metadata.savm_id == "12345"
    assert metadata.savm_name_with_id == "Example Company (12345)"
    assert metadata.sales_level_1 == "Level 1"
    assert metadata.sales_level_2 == "Level 2"
    assert metadata.sales_level_3 == "Level 3"
    assert metadata.sales_level_4 == "Level 4"
    assert metadata.additional_metadata == {"custom_field": "custom_value"}
    
    # Test to_dict method
    metadata_dict = metadata.to_dict()
    assert metadata_dict["SAVM_ID"] == "12345"
    assert metadata_dict["SAVM_NAME_WITH_ID"] == "Example Company (12345)"
    assert metadata_dict["SALES_LEVEL_1"] == "Level 1"
    assert metadata_dict["SALES_LEVEL_2"] == "Level 2"
    assert metadata_dict["SALES_LEVEL_3"] == "Level 3"
    assert metadata_dict["SALES_LEVEL_4"] == "Level 4"
    assert metadata_dict["custom_field"] == "custom_value"
    
    # Test from_dict method
    metadata2 = CompanyMetadata.from_dict(metadata_dict)
    assert metadata2.savm_id == metadata.savm_id
    assert metadata2.savm_name_with_id == metadata.savm_name_with_id
    assert metadata2.sales_level_1 == metadata.sales_level_1
    assert metadata2.sales_level_2 == metadata.sales_level_2
    assert metadata2.sales_level_3 == metadata.sales_level_3
    assert metadata2.sales_level_4 == metadata.sales_level_4
    assert metadata2.additional_metadata["custom_field"] == metadata.additional_metadata["custom_field"]

# Test Company
def test_company():
    """Test the Company class."""
    # Create a Company instance
    company = Company(
        id="abc123",
        name="Example Company",
        url="https://example.com",
        domain="example.com",
        metadata=CompanyMetadata(savm_id="12345"),
        last_updated="2023-01-01T00:00:00"
    )
    
    # Test that the attributes are set correctly
    assert company.id == "abc123"
    assert company.name == "Example Company"
    assert company.url == "https://example.com"
    assert company.domain == "example.com"
    assert company.metadata.savm_id == "12345"
    assert company.last_updated == "2023-01-01T00:00:00"
    
    # Test to_dict method
    company_dict = company.to_dict()
    assert company_dict["id"] == "abc123"
    assert company_dict["name"] == "Example Company"
    assert company_dict["url"] == "https://example.com"
    assert company_dict["domain"] == "example.com"
    assert company_dict["metadata"]["SAVM_ID"] == "12345"
    assert company_dict["last_updated"] == "2023-01-01T00:00:00"
    
    # Test from_dict method
    company2 = Company.from_dict(company_dict)
    assert company2.id == company.id
    assert company2.name == company.name
    assert company2.url == company.url
    assert company2.domain == company.domain
    assert company2.metadata.savm_id == company.metadata.savm_id
    assert company2.last_updated == company.last_updated

# Test Company.from_url
def test_company_from_url():
    """Test the Company.from_url method."""
    # Create a Company from a URL
    company = Company.from_url("https://example.com")
    
    # Test that the attributes are set correctly
    assert company.name == "Example"
    assert company.url == "https://example.com"
    assert company.domain == "example.com"
    
    # Test with a URL that has www
    company = Company.from_url("https://www.example.com")
    assert company.domain == "example.com"
    
    # Test with a URL that has a path
    company = Company.from_url("https://example.com/path/to/page")
    assert company.domain == "example.com"
    
    # Test with a URL that has a subdomain
    company = Company.from_url("https://subdomain.example.com")
    assert company.domain == "subdomain.example.com"
    
    # Test with a URL that doesn't have a protocol
    company = Company.from_url("example.com")
    assert company.domain == "example.com"
    
    # Test with metadata
    metadata = {"SAVM_ID": "12345", "SAVM_NAME_WITH_ID": "Example Company (12345)"}
    company = Company.from_url("https://example.com", metadata)
    assert company.metadata.savm_id == "12345"
    assert company.metadata.savm_name_with_id == "Example Company (12345)"

# Test Company.extract_domain
def test_extract_domain():
    """Test the Company.extract_domain method."""
    # Test with a URL that has a protocol
    domain = Company.extract_domain("https://example.com")
    assert domain == "example.com"
    
    # Test with a URL that has www
    domain = Company.extract_domain("https://www.example.com")
    assert domain == "example.com"
    
    # Test with a URL that has a path
    domain = Company.extract_domain("https://example.com/path/to/page")
    assert domain == "example.com"
    
    # Test with a URL that has a subdomain
    domain = Company.extract_domain("https://subdomain.example.com")
    assert domain == "subdomain.example.com"
    
    # Test with a URL that doesn't have a protocol
    domain = Company.extract_domain("example.com")
    assert domain == "example.com"
    
    # Test with an empty URL
    domain = Company.extract_domain("")
    assert domain == ""

# Test Company.extract_company_name
def test_extract_company_name():
    """Test the Company.extract_company_name method."""
    # Test with a simple domain
    name = Company.extract_company_name("example.com")
    assert name == "Example"
    
    # Test with a domain that has hyphens
    name = Company.extract_company_name("example-company.com")
    assert name == "Example Company"
    
    # Test with a domain that has underscores
    name = Company.extract_company_name("example_company.com")
    assert name == "Example Company"
    
    # Test with a domain that has multiple parts
    name = Company.extract_company_name("example-company-inc.com")
    assert name == "Example Company Inc"
    
    # Test with an empty domain
    name = Company.extract_company_name("")
    assert name == ""

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
