#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report Models

This module provides data models for reports and related concepts.
It includes models for reports, report sections, and report metadata.

Key classes:
- Report: Represents a report with its sections and metadata
- ReportSection: Represents a section of a report
- ReportMetadata: Represents metadata for a report
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class ReportMetadata:
    """
    Represents metadata for a report.
    
    Attributes:
        url: URL associated with the report
        topic: Topic of the report
        customer_name: Name of the customer
        company_name: Name of the company
        generation_date: Date when the report was generated
        savm_id: SAVM ID associated with the report
        additional_metadata: Additional metadata as key-value pairs
    """
    url: str = ""
    topic: str = ""
    customer_name: Optional[str] = None
    company_name: Optional[str] = None
    generation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    savm_id: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report metadata to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the report metadata
        """
        result = {
            "url": self.url,
            "topic": self.topic,
            "generation_date": self.generation_date
        }
        
        # Add optional fields if they exist
        if self.customer_name:
            result["customer_name"] = self.customer_name
        
        if self.company_name:
            result["company_name"] = self.company_name
        
        if self.savm_id:
            result["savm_id"] = self.savm_id
        
        # Add additional metadata
        result.update(self.additional_metadata)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportMetadata':
        """
        Create report metadata from a dictionary.
        
        Args:
            data: Dictionary with report metadata
            
        Returns:
            ReportMetadata: ReportMetadata instance
        """
        # Extract known fields
        url = data.get("url", "")
        topic = data.get("topic", "")
        customer_name = data.get("customer_name")
        company_name = data.get("company_name")
        generation_date = data.get("generation_date", datetime.now().isoformat())
        savm_id = data.get("savm_id")
        
        # Extract additional metadata (all other fields)
        additional_metadata = {}
        for key, value in data.items():
            if key not in ["url", "topic", "customer_name", "company_name", "generation_date", "savm_id"]:
                additional_metadata[key] = value
        
        return cls(
            url=url,
            topic=topic,
            customer_name=customer_name,
            company_name=company_name,
            generation_date=generation_date,
            savm_id=savm_id,
            additional_metadata=additional_metadata
        )

@dataclass
class ReportSection:
    """
    Represents a section of a report.
    
    Attributes:
        id: Unique identifier for the section
        level: Heading level (1 for main heading, 2 for subheading, etc.)
        title: Title of the section
        content: Content of the section
        section_type: Type of the section (introduction, company_news, etc.)
        sources: List of sources for the section
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 1
    title: str = ""
    content: str = ""
    section_type: str = "other"
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report section to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the report section
        """
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "sources": self.sources
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportSection':
        """
        Create a report section from a dictionary.
        
        Args:
            data: Dictionary with report section data
            
        Returns:
            ReportSection: ReportSection instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            level=data.get("level", 1),
            title=data.get("title", ""),
            content=data.get("content", ""),
            section_type=data.get("section_type", "other"),
            sources=data.get("sources", [])
        )

@dataclass
class Report:
    """
    Represents a report with its sections and metadata.
    
    Attributes:
        id: Unique identifier for the report
        metadata: Metadata for the report
        sections: List of report sections
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: ReportMetadata = field(default_factory=ReportMetadata)
    sections: List[ReportSection] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the report
        """
        return {
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "sections": [section.to_dict() for section in self.sections]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Report':
        """
        Create a report from a dictionary.
        
        Args:
            data: Dictionary with report data
            
        Returns:
            Report: Report instance
        """
        # Extract ID
        report_id = data.get("id", str(uuid.uuid4()))
        
        # Extract metadata
        metadata_data = data.get("metadata", {})
        metadata = ReportMetadata.from_dict(metadata_data)
        
        # Extract sections
        sections_data = data.get("sections", [])
        sections = [ReportSection.from_dict(section_data) for section_data in sections_data]
        
        return cls(
            id=report_id,
            metadata=metadata,
            sections=sections
        )
    
    def to_markdown(self) -> str:
        """
        Convert the report to markdown format.
        
        Returns:
            str: Markdown representation of the report
        """
        # Start with the title
        markdown = f"# Research Report on {self.metadata.customer_name or self.metadata.company_name or self.metadata.url}\n\n"
        
        # Add generation date
        try:
            generation_date = datetime.fromisoformat(self.metadata.generation_date)
            date_str = generation_date.strftime('%B %d, %Y')
        except:
            date_str = self.metadata.generation_date
        
        markdown += f"*Generated on {date_str}*\n\n"
        
        # Add URL
        markdown += f"## URL: {self.metadata.url}\n\n"
        
        # Add sections
        for section in sorted(self.sections, key=lambda s: (s.level, s.title)):
            # Add section heading
            markdown += f"{'#' * section.level} {section.title}\n\n"
            
            # Add section content
            markdown += f"{section.content}\n\n"
            
            # Add sources if present
            if section.sources:
                markdown += "### Sources\n\n"
                for i, source in enumerate(section.sources):
                    markdown += f"[{i+1}]: {source}\n"
                markdown += "\n"
        
        return markdown
    
    @classmethod
    def from_markdown(cls, markdown: str, url: str, topic: str, customer_name: Optional[str] = None) -> 'Report':
        """
        Create a report from markdown content.
        
        Args:
            markdown: Markdown content
            url: URL associated with the report
            topic: Topic of the report
            customer_name: Name of the customer
            
        Returns:
            Report: Report instance
        """
        from utils.file_utils import markdown_to_json
        
        # Extract company name from URL or customer name
        company_name = None
        if customer_name:
            # Try to extract company name from customer name
            if "(" in customer_name:
                company_name = customer_name.split("(")[0].strip()
            else:
                company_name = customer_name
        
        # Convert markdown to JSON
        json_data = markdown_to_json(markdown, url, topic, customer_name, company_name)
        
        # Create report from JSON
        return cls.from_dict(json_data)
