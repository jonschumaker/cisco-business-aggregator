#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Product Models

This module provides data models for products and related concepts.
It includes models for products, product analysis, and price-value assessment.

Key classes:
- Product: Represents a product with its attributes
- ProductAnalysis: Represents the analysis of a product
- PriceValueAssessment: Represents the price-value assessment of a product
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Product:
    """
    Represents a product with its attributes.
    
    Attributes:
        id: Unique identifier for the product
        name: Name of the product
        model: Model number of the product
        manufacturer: Manufacturer of the product
        category: Category of the product
        features: List of product features
        specifications: Dictionary of product specifications
        price: Price of the product as a string (e.g., "$100")
        price_numeric: Price of the product as a number
        performance: Dictionary of performance metrics
        release_date: Release date of the product
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    model: str = ""
    manufacturer: str = ""
    category: str = ""
    features: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    price: Optional[str] = None
    price_numeric: Optional[float] = None
    performance: Dict[str, Any] = field(default_factory=dict)
    release_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the product to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the product
        """
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "manufacturer": self.manufacturer,
            "category": self.category,
            "features": self.features,
            "specifications": self.specifications,
            "price": self.price,
            "price_numeric": self.price_numeric,
            "performance": self.performance,
            "release_date": self.release_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """
        Create a product from a dictionary.
        
        Args:
            data: Dictionary with product data
            
        Returns:
            Product: Product instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            model=data.get("model", ""),
            manufacturer=data.get("manufacturer", ""),
            category=data.get("category", ""),
            features=data.get("features", []),
            specifications=data.get("specifications", {}),
            price=data.get("price"),
            price_numeric=data.get("price_numeric"),
            performance=data.get("performance", {}),
            release_date=data.get("release_date")
        )

@dataclass
class ProductAnalysis:
    """
    Represents the analysis of a product.
    
    Attributes:
        feature_categorization: Categorization of features (basic, advanced, premium)
        market_positioning: Market positioning (entry-level, mid-range, high-end, enterprise)
        standout_features: List of standout features
        category_comparison: Comparison to typical features in the product category
        competitive_positioning: Competitive positioning relative to similar products
    """
    feature_categorization: str = ""
    market_positioning: str = ""
    standout_features: List[str] = field(default_factory=list)
    category_comparison: str = ""
    competitive_positioning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the product analysis to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the product analysis
        """
        return {
            "feature_categorization": self.feature_categorization,
            "market_positioning": self.market_positioning,
            "standout_features": self.standout_features,
            "category_comparison": self.category_comparison,
            "competitive_positioning": self.competitive_positioning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductAnalysis':
        """
        Create a product analysis from a dictionary.
        
        Args:
            data: Dictionary with product analysis data
            
        Returns:
            ProductAnalysis: ProductAnalysis instance
        """
        return cls(
            feature_categorization=data.get("feature_categorization", ""),
            market_positioning=data.get("market_positioning", ""),
            standout_features=data.get("standout_features", []),
            category_comparison=data.get("category_comparison", ""),
            competitive_positioning=data.get("competitive_positioning", "")
        )

@dataclass
class PriceValueAssessment:
    """
    Represents the price-value assessment of a product.
    
    Attributes:
        feature_to_price_ratio: Feature-to-price ratio assessment
        performance_to_price_ratio: Performance-to-price ratio assessment
        market_position_gap: Market position gap assessment
        total_cost_of_ownership: Total cost of ownership assessment
        innovation_premium: Innovation premium assessment
        overall_score: Overall price-value score
        underpriced_percentage: Percentage by which the product is underpriced
        assessment: Overall assessment explanation
    """
    feature_to_price_ratio: Dict[str, Any] = field(default_factory=lambda: {"score": 0, "explanation": ""})
    performance_to_price_ratio: Dict[str, Any] = field(default_factory=lambda: {"score": 0, "explanation": ""})
    market_position_gap: Dict[str, Any] = field(default_factory=lambda: {"score": 0, "explanation": ""})
    total_cost_of_ownership: Dict[str, Any] = field(default_factory=lambda: {"score": 0, "explanation": ""})
    innovation_premium: Dict[str, Any] = field(default_factory=lambda: {"score": 0, "explanation": ""})
    overall_score: float = 0.0
    underpriced_percentage: float = 0.0
    assessment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the price-value assessment to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the price-value assessment
        """
        return {
            "feature_to_price_ratio": self.feature_to_price_ratio,
            "performance_to_price_ratio": self.performance_to_price_ratio,
            "market_position_gap": self.market_position_gap,
            "total_cost_of_ownership": self.total_cost_of_ownership,
            "innovation_premium": self.innovation_premium,
            "overall_score": self.overall_score,
            "underpriced_percentage": self.underpriced_percentage,
            "assessment": self.assessment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceValueAssessment':
        """
        Create a price-value assessment from a dictionary.
        
        Args:
            data: Dictionary with price-value assessment data
            
        Returns:
            PriceValueAssessment: PriceValueAssessment instance
        """
        return cls(
            feature_to_price_ratio=data.get("feature_to_price_ratio", {"score": 0, "explanation": ""}),
            performance_to_price_ratio=data.get("performance_to_price_ratio", {"score": 0, "explanation": ""}),
            market_position_gap=data.get("market_position_gap", {"score": 0, "explanation": ""}),
            total_cost_of_ownership=data.get("total_cost_of_ownership", {"score": 0, "explanation": ""}),
            innovation_premium=data.get("innovation_premium", {"score": 0, "explanation": ""}),
            overall_score=data.get("overall_score", 0.0),
            underpriced_percentage=data.get("underpriced_percentage", 0.0),
            assessment=data.get("assessment", "")
        )

@dataclass
class AnalyzedProduct(Product):
    """
    Represents a product with analysis results.
    
    This class extends the Product class with analysis results.
    
    Attributes:
        analysis: Analysis results for the product
        price_value_assessment: Price-value assessment for the product
    """
    analysis: ProductAnalysis = field(default_factory=ProductAnalysis)
    price_value_assessment: PriceValueAssessment = field(default_factory=PriceValueAssessment)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the analyzed product to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the analyzed product
        """
        result = super().to_dict()
        result.update({
            "analysis": self.analysis.to_dict(),
            "price_value_assessment": self.price_value_assessment.to_dict()
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyzedProduct':
        """
        Create an analyzed product from a dictionary.
        
        Args:
            data: Dictionary with analyzed product data
            
        Returns:
            AnalyzedProduct: AnalyzedProduct instance
        """
        product = Product.from_dict(data)
        
        analysis_data = data.get("analysis", {})
        if isinstance(analysis_data, dict):
            analysis = ProductAnalysis.from_dict(analysis_data)
        else:
            analysis = ProductAnalysis()
        
        assessment_data = data.get("price_value_assessment", {})
        if isinstance(assessment_data, dict):
            assessment = PriceValueAssessment.from_dict(assessment_data)
        else:
            assessment = PriceValueAssessment()
        
        return cls(
            id=product.id,
            name=product.name,
            model=product.model,
            manufacturer=product.manufacturer,
            category=product.category,
            features=product.features,
            specifications=product.specifications,
            price=product.price,
            price_numeric=product.price_numeric,
            performance=product.performance,
            release_date=product.release_date,
            analysis=analysis,
            price_value_assessment=assessment
        )
