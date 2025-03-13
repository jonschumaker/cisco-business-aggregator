#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Company Models

This module provides data models for companies and related concepts.
It includes models for companies, company metadata, and company analysis.

Key classes:
- Company: Represents a company with its attributes
- CompanyMetadata: Represents metadata for a company
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
from urllib.parse import urlparse

@dataclass
class CompanyMetadata:
    """
    Represents metadata for a company.
    
    Attributes:
        savm_id: SAVM ID associated with the company
        savm_name_with_id: SAVM name with ID
        sales_level_1: Sales level 1
        sales_level_2: Sales level 2
        sales_level_3: Sales level 3
        sales_level_4: Sales level 4
        additional_metadata: Additional metadata as key-value pairs
    """
    savm_id: Optional[str] = None
    savm_name_with_id: Optional[str] = None
    sales_level_1: Optional[str] = None
    sales_level_2: Optional[str] = None
    sales_level_3: Optional[str] = None
    sales_level_4: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the company metadata to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the company metadata
        """
        result = {}
        
        # Add fields if they exist
        if self.savm_id:
            result["SAVM_ID"] = self.savm_id
        
        if self.savm_name_with_id:
            result["SAVM_NAME_WITH_ID"] = self.savm_name_with_id
        
        if self.sales_level_1:
            result["SALES_LEVEL_1"] = self.sales_level_1
        
        if self.sales_level_2:
            result["SALES_LEVEL_2"] = self.sales_level_2
        
        if self.sales_level_3:
            result["SALES_LEVEL_3"] = self.sales_level_3
        
        if self.sales_level_4:
            result["SALES_LEVEL_4"] = self.sales_level_4
        
        # Add additional metadata
        result.update(self.additional_metadata)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompanyMetadata':
        """
        Create company metadata from a dictionary.
        
        Args:
            data: Dictionary with company metadata
            
        Returns:
            CompanyMetadata: CompanyMetadata instance
        """
        # Extract known fields
        savm_id = data.get("SAVM_ID")
        savm_name_with_id = data.get("SAVM_NAME_WITH_ID")
        sales_level_1 = data.get("SALES_LEVEL_1")
        sales_level_2 = data.get("SALES_LEVEL_2")
        sales_level_3 = data.get("SALES_LEVEL_3")
        sales_level_4 = data.get("SALES_LEVEL_4")
        
        # Extract additional metadata (all other fields)
        additional_metadata = {}
        for key, value in data.items():
            if key not in ["SAVM_ID", "SAVM_NAME_WITH_ID", "SALES_LEVEL_1", "SALES_LEVEL_2", "SALES_LEVEL_3", "SALES_LEVEL_4"]:
                additional_metadata[key] = value
        
        return cls(
            savm_id=savm_id,
            savm_name_with_id=savm_name_with_id,
            sales_level_1=sales_level_1,
            sales_level_2=sales_level_2,
            sales_level_3=sales_level_3,
            sales_level_4=sales_level_4,
            additional_metadata=additional_metadata
        )

@dataclass
class Company:
    """
    Represents a company with its attributes.
    
    Attributes:
        id: Unique identifier for the company
        name: Name of the company
        url: URL of the company website
        domain: Domain of the company website
        metadata: Metadata for the company
        last_updated: Date when the company information was last updated
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    url: str = ""
    domain: str = ""
    metadata: CompanyMetadata = field(default_factory=CompanyMetadata)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """
        Post-initialization processing.
        
        This method is called after the object is initialized.
        It extracts the domain from the URL if the domain is not provided.
        """
        if self.url and not self.domain:
            self.domain = self.extract_domain(self.url)
        
        if not self.name and self.metadata.savm_name_with_id:
            # Extract name from SAVM_NAME_WITH_ID
            if "(" in self.metadata.savm_name_with_id:
                self.name = self.metadata.savm_name_with_id.split("(")[0].strip()
            else:
                self.name = self.metadata.savm_name_with_id
        elif not self.name and self.domain:
            # Extract name from domain
            self.name = self.extract_company_name(self.domain)
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract the domain from a URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            str: Domain extracted from the URL
        """
        if not url:
            return ""
        
        # Add https:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
        
        try:
            # Use standard URL parsing
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except:
            # Fallback method if standard parsing fails
            try:
                # Simple approach: remove protocols and split by first /
                url = url.lower()
                url = url.replace('https://', '').replace('http://', '')
                if url.startswith('www.'):
                    url = url[4:]
                return url.split('/')[0]
            except:
                return ""
    
    @staticmethod
    def extract_company_name(domain: str) -> str:
        """
        Extract a clean company name from a domain.
        
        Args:
            domain: Domain to extract company name from
            
        Returns:
            str: Company name extracted from the domain
        """
        if not domain:
            return ""
        
        # Get the first part of the domain (before the first dot)
        company = domain.split('.')[0]
        
        # Clean up and capitalize
        company = company.replace('-', ' ').replace('_', ' ')
        company = ' '.join(word.capitalize() for word in company.split())
        
        return company
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the company to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the company
        """
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "domain": self.domain,
            "metadata": self.metadata.to_dict(),
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """
        Create a company from a dictionary.
        
        Args:
            data: Dictionary with company data
            
        Returns:
            Company: Company instance
        """
        # Extract metadata
        metadata_data = data.get("metadata", {})
        metadata = CompanyMetadata.from_dict(metadata_data)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            metadata=metadata,
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )
    
    @classmethod
    def from_url(cls, url: str, metadata: Optional[Dict[str, Any]] = None) -> 'Company':
        """
        Create a company from a URL.
        
        Args:
            url: URL of the company website
            metadata: Optional metadata for the company
            
        Returns:
            Company: Company instance
        """
        # Extract domain from URL
        domain = cls.extract_domain(url)
        
        # Extract company name from domain
        name = cls.extract_company_name(domain)
        
        # Create metadata object
        if metadata:
            company_metadata = CompanyMetadata.from_dict(metadata)
        else:
            company_metadata = CompanyMetadata()
        
        return cls(
            name=name,
            url=url,
            domain=domain,
            metadata=company_metadata
        )
