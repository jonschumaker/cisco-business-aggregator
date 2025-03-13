#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Product Innovation Agent: Underpriced Products in Enterprise Networking Manufacturing

This module identifies underpriced products in the enterprise computer/networking manufacturing space.
It analyzes products across various categories including computer hardware, cybersecurity, firewalls,
networking equipment, switching, cyber defense, and computer parts manufacturing.

Key Features:
- Data collection from manufacturer websites, industry databases, and product reviews
- Multi-factor analysis of product features, performance, and pricing
- Price-value assessment to identify underpriced products
- Percentage-based scoring of how underpriced products are
- Integration with LangGraph for orchestrating the analysis workflow
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from urllib.parse import urlparse
import uuid
from dotenv import load_dotenv
from utils.cache import cache_tavily_search, cache_llm_call, async_cache_with_ttl
from config import settings
# Get cache TTL values from settings
CACHE_TTL_SEARCH = getattr(settings, 'CACHE_TTL_SEARCH', 3600)
CACHE_TTL_LLM = getattr(settings, 'CACHE_TTL_LLM', 3600)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("product_innovation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Azure OpenAI environment variables with the correct names
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "https://phx-sales-ai.openai.azure.com/")
os.environ["AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Import for Tavily search
from tavily import TavilyClient

# Import for LangGraph
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Initialize API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API keys are available
if not TAVILY_API_KEY:
    logger.error("TAVILY_API_KEY not found in environment variables.")
    raise ValueError("Missing TAVILY_API_KEY in environment variables")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Set environment variables for libraries that need them
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Define product categories to focus on
PRODUCT_CATEGORIES = [
    "enterprise computer hardware",
    "cybersecurity appliances",
    "firewall systems",
    "networking equipment",
    "network switches",
    "cyber defense solutions",
    "enterprise computer parts"
]

# Define manufacturers to analyze (can be expanded)
TARGET_MANUFACTURERS = [
    "Cisco",
    "Juniper Networks",
    "Palo Alto Networks",
    "Fortinet",
    "Arista Networks",
    "HPE",
    "Dell",
    "IBM",
    "Huawei",
    "Check Point",
    "SonicWall",
    "Ubiquiti",
    "Netgear",
    "Aruba Networks",
    "F5 Networks"
]

# Define the criteria for determining if a product is underpriced
UNDERPRICED_CRITERIA = {
    "feature_to_price_ratio": {
        "description": "Measures the number and quality of features relative to price point",
        "weight": 0.25
    },
    "performance_to_price_ratio": {
        "description": "Evaluates performance metrics against cost",
        "weight": 0.25
    },
    "market_position_gap": {
        "description": "Identifies products with capabilities of higher-tier products but priced in lower tiers",
        "weight": 0.20
    },
    "total_cost_of_ownership": {
        "description": "Factors in operational costs, maintenance, and lifespan",
        "weight": 0.15
    },
    "innovation_premium": {
        "description": "Assesses whether innovative features are appropriately priced into the product",
        "weight": 0.15
    }
}

class ProductInnovationAgent:
    """
    Agent for identifying underpriced products in enterprise networking manufacturing.
    
    This class orchestrates the process of collecting data, analyzing products,
    assessing price-value relationships, and identifying underpriced products.
    """
    
    @cache_tavily_search
    async def _cached_tavily_search(self, **kwargs):
        """
        Cached wrapper for Tavily search API.
        
        This method wraps the Tavily search API call with caching to reduce redundant API calls.
        
        Args:
            **kwargs: The parameters to pass to the Tavily search API
            
        Returns:
            The search results from Tavily
        """
        logger.info(f"Performing Tavily search with parameters: {kwargs}")
        return self.tavily_client.search(**kwargs)
    
    def __init__(self, llm_model="gpt-4o", llm_provider="azure_openai"):
        """Initialize the Product Innovation Agent."""
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.1,
            model_kwargs={
                "azure_deployment": os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://phx-sales-ai.openai.azure.com/")
            }
        )
        
        # Initialize memory for the LangGraph
        self.memory = MemorySaver()
        
        # Build the LangGraph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for the product innovation workflow.
        
        The graph consists of the following nodes:
        1. Data Collection: Gathers information about products
        2. Product Analysis: Extracts features and performance metrics
        3. Price-Value Assessment: Evaluates pricing against value metrics
        4. Underpriced Product Identification: Identifies underpriced products
        5. Report Generation: Creates the final report
        
        Returns:
            StateGraph: The compiled LangGraph
        """
        # Define the state schema
        from typing import TypedDict, List, Dict, Any, Optional
        
        class ProductData(TypedDict, total=False):
            id: str
            name: str
            model: str
            manufacturer: str
            category: str
            features: List[str]
            specifications: Dict[str, Any]
            price: Optional[str]
            price_numeric: Optional[float]
            performance: Dict[str, Any]
            release_date: Optional[str]
        
        class AnalyzedProduct(ProductData, total=False):
            feature_categorization: str
            market_positioning: str
            standout_features: List[str]
            category_comparison: str
            competitive_positioning: str
        
        class AssessedProduct(AnalyzedProduct, total=False):
            price_value_assessment: Dict[str, Any]
        
        class ProductInnovationState(TypedDict, total=False):
            category: Optional[str]
            manufacturer: Optional[str]
            product_data: List[ProductData]
            analyzed_products: List[AnalyzedProduct]
            assessed_products: List[AssessedProduct]
            underpriced_products: List[AssessedProduct]
            underpriced_by_category: Dict[str, List[AssessedProduct]]
            underpriced_by_manufacturer: Dict[str, List[AssessedProduct]]
            report: Dict[str, Any]
        
        # Define the graph
        builder = StateGraph(ProductInnovationState)
        
        # Add nodes to the graph
        builder.add_node("data_collection", self._data_collection_node)
        builder.add_node("product_analysis", self._product_analysis_node)
        builder.add_node("price_value_assessment", self._price_value_assessment_node)
        builder.add_node("underpriced_identification", self._underpriced_identification_node)
        builder.add_node("report_generation", self._report_generation_node)
        
        # Define the edges (workflow)
        builder.add_edge("data_collection", "product_analysis")
        builder.add_edge("product_analysis", "price_value_assessment")
        builder.add_edge("price_value_assessment", "underpriced_identification")
        builder.add_edge("underpriced_identification", "report_generation")
        builder.add_edge("report_generation", END)
        
        # Set the entry point
        builder.set_entry_point("data_collection")
        
        # Compile the graph
        return builder.compile()
    
    async def _data_collection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect data about products from various sources.
        
        This node:
        1. Searches for products in the specified categories
        2. Collects information about features, specifications, and pricing
        3. Organizes the data for further analysis
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with collected product data
        """
        logger.info("Starting data collection for product innovation analysis")
        
        # Extract parameters from state
        category = state.get("category", None)
        manufacturer = state.get("manufacturer", None)
        
        # If no specific category or manufacturer is provided, use defaults
        categories_to_search = [category] if category else PRODUCT_CATEGORIES
        manufacturers_to_search = [manufacturer] if manufacturer else TARGET_MANUFACTURERS
        
        # Initialize product data storage
        product_data = []
        
        # Search for products in each category and manufacturer combination
        for category in categories_to_search:
            for manufacturer in manufacturers_to_search:
                logger.info(f"Searching for {manufacturer} products in {category} category")
                
                # Construct search query
                query = f"{manufacturer} {category} products pricing specifications"
                
                try:
                    # Perform search using Tavily with caching
                    search_results = await self._cached_tavily_search(
                        query=query,
                        search_depth="advanced",
                        include_answer=True,
                        include_domains=None,
                        exclude_domains=None
                    )
                    
                    # Extract relevant information from search results
                    if "answer" in search_results and search_results["answer"]:
                        # Use the LLM to extract structured product information from the answer
                        extracted_products = await self._extract_product_info(
                            search_results["answer"], 
                            category, 
                            manufacturer
                        )
                        product_data.extend(extracted_products)
                    
                    # Also process the context sections for additional information
                    if "context" in search_results and search_results["context"]:
                        for context_item in search_results["context"]:
                            if "content" in context_item and context_item["content"]:
                                # Extract product information from context
                                extracted_products = await self._extract_product_info(
                                    context_item["content"], 
                                    category, 
                                    manufacturer
                                )
                                product_data.extend(extracted_products)
                except Exception as e:
                    logger.error(f"Error searching for {manufacturer} {category} products: {str(e)}")
        
        # Deduplicate products based on name and model number
        unique_products = self._deduplicate_products(product_data)
        
        logger.info(f"Collected information on {len(unique_products)} unique products")
        
        # Update state with collected product data
        return {
            **state,
            "product_data": unique_products
        }
    
    @cache_llm_call
    async def _extract_product_info(self, text: str, category: str, manufacturer: str) -> List[Dict[str, Any]]:
        """
        Extract structured product information from text using the LLM.
        
        Args:
            text: The text containing product information
            category: The product category
            manufacturer: The manufacturer name
            
        Returns:
            List of extracted product information dictionaries
        """
        # Construct the prompt for the LLM
        prompt = f"""
        Extract information about {manufacturer} products in the {category} category from the following text.
        For each product mentioned, extract:
        1. Product name and model number
        2. Key features and specifications
        3. Price information (if available)
        4. Performance metrics (if available)
        5. Release date or generation (if available)
        
        Format the information as a list of JSON objects, with each object representing a product.
        If no specific products are mentioned, return an empty list.
        
        Text to analyze:
        {text}
        """
        
        # Call the LLM to extract product information
        response = await self.llm.ainvoke(prompt)
        
        # Parse the response to extract product information
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to find array directly
                json_str = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL)
                if json_str:
                    json_str = json_str.group(0)
                else:
                    # If still no match, use the entire response
                    json_str = response.content
            
            # Parse the JSON string
            products = json.loads(json_str)
            
            # Ensure the result is a list
            if not isinstance(products, list):
                products = [products]
            
            # Add category and manufacturer to each product
            for product in products:
                product["category"] = category
                product["manufacturer"] = manufacturer
                product["id"] = str(uuid.uuid4())  # Add a unique ID
            
            return products
        except Exception as e:
            logger.error(f"Error parsing product information: {str(e)}")
            logger.error(f"LLM response: {response.content}")
            return []
    
    def _deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate products based on name and model number.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Deduplicated list of products
        """
        unique_products = {}
        
        for product in products:
            # Create a key based on manufacturer and product name/model
            product_name = product.get("product_name", "").lower() if product.get("product_name") else ""
            model_number = product.get("model_number", "").lower() if product.get("model_number") else ""
            manufacturer = product.get("manufacturer", "").lower()
            
            # If product has both name and model, use both for the key
            if product_name and model_number:
                key = f"{manufacturer}_{product_name}_{model_number}"
            # If only one is available, use what's available
            elif product_name:
                key = f"{manufacturer}_{product_name}"
            elif model_number:
                key = f"{manufacturer}_{model_number}"
            else:
                # Skip products without identifiable information
                continue
            
            # If this is a new product or has more information than the existing one, keep it
            if key not in unique_products or len(str(product)) > len(str(unique_products[key])):
                unique_products[key] = product
        
        return list(unique_products.values())
    
    async def _product_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze products to extract features, performance metrics, and other relevant information.
        
        This node:
        1. Processes the collected product data
        2. Extracts and categorizes features
        3. Identifies performance metrics
        4. Analyzes competitive positioning
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with analyzed product data
        """
        logger.info("Starting product analysis")
        
        # Get product data from state
        product_data = state.get("product_data", [])
        
        if not product_data:
            logger.warning("No product data available for analysis")
            return {
                **state,
                "analyzed_products": []
            }
        
        # Initialize storage for analyzed products
        analyzed_products = []
        
        # Analyze each product
        for product in product_data:
            try:
                # Extract product details for analysis
                product_details = {
                    "id": product.get("id", str(uuid.uuid4())),
                    "name": product.get("product_name", "Unknown"),
                    "model": product.get("model_number", "Unknown"),
                    "manufacturer": product.get("manufacturer", "Unknown"),
                    "category": product.get("category", "Unknown"),
                    "features": product.get("key_features", []),
                    "specifications": product.get("specifications", {}),
                    "price": product.get("price", None),
                    "performance": product.get("performance_metrics", {}),
                    "release_date": product.get("release_date", None)
                }
                
                # If price is a string, try to extract numeric value
                if isinstance(product_details["price"], str):
                    price_str = product_details["price"]
                    # Extract numeric value using regex
                    price_match = re.search(r'[\$£€]?\s*([0-9,]+(?:\.[0-9]+)?)', price_str)
                    if price_match:
                        # Convert to float, removing commas
                        try:
                            numeric_price = float(price_match.group(1).replace(',', ''))
                            product_details["price_numeric"] = numeric_price
                        except ValueError:
                            product_details["price_numeric"] = None
                    else:
                        product_details["price_numeric"] = None
                
                # Use LLM to analyze the product features and competitive positioning
                analysis_result = await self._analyze_product_features(product_details)
                
                # Combine original details with analysis results
                analyzed_product = {
                    **product_details,
                    **analysis_result
                }
                
                analyzed_products.append(analyzed_product)
                
            except Exception as e:
                logger.error(f"Error analyzing product {product.get('product_name', 'Unknown')}: {str(e)}")
        
        logger.info(f"Completed analysis of {len(analyzed_products)} products")
        
        # Update state with analyzed products
        return {
            **state,
            "analyzed_products": analyzed_products
        }
    
    @cache_llm_call
    async def _analyze_product_features(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze product features and competitive positioning using the LLM.
        
        Args:
            product: Product details dictionary
            
        Returns:
            Dictionary with analysis results
        """
        # Construct the prompt for the LLM
        prompt = f"""
        Analyze the following {product["manufacturer"]} {product["category"]} product:
        
        Product Name: {product["name"]}
        Model: {product["model"]}
        Features: {product["features"]}
        Specifications: {product["specifications"]}
        Price: {product["price"]}
        Performance Metrics: {product["performance"]}
        
        Please provide:
        1. A categorization of features (basic, advanced, premium)
        2. An assessment of the product's market positioning (entry-level, mid-range, high-end, enterprise)
        3. Identification of any standout features that provide exceptional value
        4. Comparison to typical features in this product category
        5. Estimated competitive positioning relative to similar products
        
        Format your response as a JSON object with the following keys:
        - feature_categorization
        - market_positioning
        - standout_features
        - category_comparison
        - competitive_positioning
        """
        
        # Call the LLM to analyze the product
        response = await self.llm.ainvoke(prompt)
        
        # Parse the response to extract analysis results
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to find object directly
                json_str = re.search(r'\{\s*".*"\s*:.*\}', response.content, re.DOTALL)
                if json_str:
                    json_str = json_str.group(0)
                else:
                    # If still no match, use the entire response
                    json_str = response.content
            
            # Parse the JSON string
            analysis_result = json.loads(json_str)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error parsing product analysis: {str(e)}")
            logger.error(f"LLM response: {response.content}")
            return {
                "feature_categorization": "Unknown",
                "market_positioning": "Unknown",
                "standout_features": [],
                "category_comparison": "No comparison available",
                "competitive_positioning": "Unknown"
            }
    
    async def _price_value_assessment_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the price-value relationship for each product.
        
        This node:
        1. Evaluates each product against the underpriced criteria
        2. Calculates scores for each criterion
        3. Computes an overall price-value score
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with price-value assessments
        """
        logger.info("Starting price-value assessment")
        
        # Get analyzed products from state
        analyzed_products = state.get("analyzed_products", [])
        
        if not analyzed_products:
            logger.warning("No analyzed products available for price-value assessment")
            return {
                **state,
                "assessed_products": []
            }
        
        # Initialize storage for assessed products
        assessed_products = []
        
        # Assess each product
        for product in analyzed_products:
            try:
                # Use LLM to assess the product against underpriced criteria
                assessment_result = await self._assess_price_value(product)
                
                # Combine original details with assessment results
                assessed_product = {
                    **product,
                    "price_value_assessment": assessment_result
                }
                
                assessed_products.append(assessed_product)
                
            except Exception as e:
                logger.error(f"Error assessing product {product.get('name', 'Unknown')}: {str(e)}")
        
        logger.info(f"Completed price-value assessment of {len(assessed_products)} products")
        
        # Update state with assessed products
        return {
            **state,
            "assessed_products": assessed_products
        }
    
    @cache_llm_call
    async def _assess_price_value(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the price-value relationship for a product using the LLM.
        
        Args:
            product: Product details dictionary with analysis results
            
        Returns:
            Dictionary with price-value assessment results
        """
        # Construct the prompt for the LLM
        prompt = f"""
        Assess the price-value relationship for the following {product["manufacturer"]} {product["category"]} product:
        
        Product Name: {product["name"]}
        Model: {product["model"]}
        Price: {product["price"]}
        Features: {product["features"]}
        Market Positioning: {product.get("market_positioning", "Unknown")}
        Standout Features: {product.get("standout_features", [])}
        Category Comparison: {product.get("category_comparison", "Unknown")}
        Competitive Positioning: {product.get("competitive_positioning", "Unknown")}
        
        Evaluate this product against the following criteria for determining if a product is underpriced:
        
        1. Feature-to-Price Ratio (25%): Measures the number and quality of features relative to price point
        2. Performance-to-Price Ratio (25%): Evaluates performance metrics against cost
        3. Market Position Gap (20%): Identifies if the product has capabilities of higher-tier products but is priced in lower tiers
        4. Total Cost of Ownership (15%): Factors in operational costs, maintenance, and lifespan
        5. Innovation Premium (15%): Assesses whether innovative features are appropriately priced into the product
        
        For each criterion, provide:
        1. A score from 0-100 (where higher scores indicate the product is more underpriced)
        2. A brief explanation for the score
        
        Also provide an overall assessment of whether the product is underpriced, and by what percentage.
        
        Format your response as a JSON object with the following keys:
        - feature_to_price_ratio: {{"score": number, "explanation": string}}
        - performance_to_price_ratio: {{"score": number, "explanation": string}}
        - market_position_gap: {{"score": number, "explanation": string}}
        - total_cost_of_ownership: {{"score": number, "explanation": string}}
        - innovation_premium: {{"score": number, "explanation": string}}
        - overall_score: number (weighted average of the above scores)
        - underpriced_percentage: number (how underpriced the product is as a percentage)
        - assessment: string (overall assessment explanation)
        """
        
        # Call the LLM to assess the product
        response = await self.llm.ainvoke(prompt)
        
        # Parse the response to extract assessment results
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to find object directly
                json_str = re.search(r'\{\s*".*"\s*:.*\}', response.content, re.DOTALL)
                if json_str:
                    json_str = json_str.group(0)
                else:
                    # If still no match, use the entire response
                    json_str = response.content
            
            # Parse the JSON string
            assessment_result = json.loads(json_str)
            
            # Calculate overall score if not provided
            if "overall_score" not in assessment_result:
                # Calculate weighted average of criterion scores
                overall_score = 0
                for criterion, details in UNDERPRICED_CRITERIA.items():
                    if criterion in assessment_result and "score" in assessment_result[criterion]:
                        overall_score += assessment_result[criterion]["score"] * details["weight"]
                
                assessment_result["overall_score"] = overall_score
            
            # Calculate underpriced percentage if not provided
            if "underpriced_percentage" not in assessment_result:
                # Use overall score as a basis for underpriced percentage
                # A score of 70+ indicates the product is underpriced
                if assessment_result["overall_score"] >= 70:
                    underpriced_pct = (assessment_result["overall_score"] - 70) * 2
                    assessment_result["underpriced_percentage"] = min(underpriced_pct, 100)
                else:
                    assessment_result["underpriced_percentage"] = 0
            
            return assessment_result
        except Exception as e:
            logger.error(f"Error parsing price-value assessment: {str(e)}")
            logger.error(f"LLM response: {response.content}")
            return {
                "feature_to_price_ratio": {"score": 0, "explanation": "Assessment failed"},
                "performance_to_price_ratio": {"score": 0, "explanation": "Assessment failed"},
                "market_position_gap": {"score": 0, "explanation": "Assessment failed"},
                "total_cost_of_ownership": {"score": 0, "explanation": "Assessment failed"},
                "innovation_premium": {"score": 0, "explanation": "Assessment failed"},
                "overall_score": 0,
                "underpriced_percentage": 0,
                "assessment": "Assessment failed due to an error"
            }
    
    async def _underpriced_identification_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify underpriced products based on the price-value assessments.
        
        This node:
        1. Filters products based on underpriced percentage
        2. Ranks products by how underpriced they are
        3. Groups products by category and manufacturer
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with identified underpriced products
        """
        logger.info("Identifying underpriced products")
        
        # Get assessed products from state
        assessed_products = state.get("assessed_products", [])
        
        if not assessed_products:
            logger.warning("No assessed products available for underpriced identification")
            return {
                **state,
                "underpriced_products": [],
                "underpriced_by_category": {},
                "underpriced_by_manufacturer": {}
            }
        
        # Filter for underpriced products (those with underpriced percentage > 0)
        underpriced_products = [
            product for product in assessed_products
            if product.get("price_value_assessment", {}).get("underpriced_percentage", 0) > 0
        ]
        
        # Sort by underpriced percentage (descending)
        underpriced_products.sort(
            key=lambda p: p.get("price_value_assessment", {}).get("underpriced_percentage", 0),
            reverse=True
        )
        
        # Group by category
        underpriced_by_category = {}
        for product in underpriced_products:
            category = product.get("category", "Unknown")
            if category not in underpriced_by_category:
                underpriced_by_category[category] = []
            underpriced_by_category[category].append(product)
        
        # Group by manufacturer
        underpriced_by_manufacturer = {}
        for product in underpriced_products:
            manufacturer = product.get("manufacturer", "Unknown")
            if manufacturer not in underpriced_by_manufacturer:
                underpriced_by_manufacturer[manufacturer] = []
            underpriced_by_manufacturer[manufacturer].append(product)
        
        logger.info(f"Identified {len(underpriced_products)} underpriced products")
        
        # Update state with underpriced products
        return {
            **state,
            "underpriced_products": underpriced_products,
            "underpriced_by_category": underpriced_by_category,
            "underpriced_by_manufacturer": underpriced_by_manufacturer
        }
    
    async def _report_generation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report on underpriced products.
        
        This node:
        1. Creates a summary of findings
        2. Generates detailed information about underpriced products
        3. Provides insights and recommendations
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state with the generated report
        """
        logger.info("Generating report on underpriced products")
        
        # Get underpriced products from state
        underpriced_products = state.get("underpriced_products", [])
        underpriced_by_category = state.get("underpriced_by_category", {})
        underpriced_by_manufacturer = state.get("underpriced_by_manufacturer", {})
        
        if not underpriced_products:
            logger.warning("No underpriced products available for report generation")
            report = {
                "summary": "No underpriced products were identified in the analysis.",
                "products": [],
                "categories": {},
                "manufacturers": {},
                "insights": "No insights available as no underpriced products were found.",
                "timestamp": datetime.now().isoformat()
            }
            return {
                **state,
                "report": report
            }
        
        # Calculate overall statistics
        total_products = len(state.get("assessed_products", []))
        underpriced_count = len(underpriced_products)
        average_underpriced_pct = sum(
            p.get("price_value_assessment", {}).get("underpriced_percentage", 0)
            for p in underpriced_products
        ) / underpriced_count if underpriced_count > 0 else 0
        
        # Prepare category summaries
        category_summaries = {}
        for category, products in underpriced_by_category.items():
            category_avg_pct = sum(
                p.get("price_value_assessment", {}).get("underpriced_percentage", 0)
                for p in products
            ) / len(products) if products else 0
            
            category_summaries[category] = {
                "count": len(products),
                "average_underpriced_percentage": category_avg_pct,
                "top_products": [
                    {
                        "name": p.get("name", "Unknown"),
                        "model": p.get("model", "Unknown"),
                        "manufacturer": p.get("manufacturer", "Unknown"),
                        "underpriced_percentage": p.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                        "assessment": p.get("price_value_assessment", {}).get("assessment", "No assessment available")
                    }
                    for p in sorted(
                        products,
                        key=lambda x: x.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                        reverse=True
                    )[:3]  # Top 3 products in each category
                ]
            }
        
        # Prepare manufacturer summaries
        manufacturer_summaries = {}
        for manufacturer, products in underpriced_by_manufacturer.items():
            manufacturer_avg_pct = sum(
                p.get("price_value_assessment", {}).get("underpriced_percentage", 0)
                for p in products
            ) / len(products) if products else 0
            
            manufacturer_summaries[manufacturer] = {
                "count": len(products),
                "average_underpriced_percentage": manufacturer_avg_pct,
                "top_products": [
                    {
                        "name": p.get("name", "Unknown"),
                        "model": p.get("model", "Unknown"),
                        "category": p.get("category", "Unknown"),
                        "underpriced_percentage": p.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                        "assessment": p.get("price_value_assessment", {}).get("assessment", "No assessment available")
                    }
                    for p in sorted(
                        products,
                        key=lambda x: x.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                        reverse=True
                    )[:3]  # Top 3 products for each manufacturer
                ]
            }
        
        # Generate insights using LLM
        insights = await self._generate_insights(
            underpriced_products,
            category_summaries,
            manufacturer_summaries,
            average_underpriced_pct
        )
        
        # Create the final report
        report = {
            "summary": {
                "total_products_analyzed": total_products,
                "underpriced_products_count": underpriced_count,
                "underpriced_products_percentage": (underpriced_count / total_products * 100) if total_products > 0 else 0,
                "average_underpriced_percentage": average_underpriced_pct
            },
            "top_underpriced_products": [
                {
                    "name": p.get("name", "Unknown"),
                    "model": p.get("model", "Unknown"),
                    "manufacturer": p.get("manufacturer", "Unknown"),
                    "category": p.get("category", "Unknown"),
                    "underpriced_percentage": p.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                    "assessment": p.get("price_value_assessment", {}).get("assessment", "No assessment available")
                }
                for p in underpriced_products[:10]  # Top 10 underpriced products overall
            ],
            "categories": category_summaries,
            "manufacturers": manufacturer_summaries,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Report generation completed")
        
        # Update state with the generated report
        return {
            **state,
            "report": report
        }
    
    @cache_llm_call
    async def _generate_insights(
        self,
        underpriced_products: List[Dict[str, Any]],
        category_summaries: Dict[str, Any],
        manufacturer_summaries: Dict[str, Any],
        average_underpriced_pct: float
    ) -> str:
        """
        Generate insights about underpriced products using the LLM.
        
        Args:
            underpriced_products: List of underpriced products
            category_summaries: Summaries by category
            manufacturer_summaries: Summaries by manufacturer
            average_underpriced_pct: Average underpriced percentage
            
        Returns:
            String with insights and recommendations
        """
        # Prepare data for the LLM
        top_products = [
            {
                "name": p.get("name", "Unknown"),
                "model": p.get("model", "Unknown"),
                "manufacturer": p.get("manufacturer", "Unknown"),
                "category": p.get("category", "Unknown"),
                "underpriced_percentage": p.get("price_value_assessment", {}).get("underpriced_percentage", 0),
                "assessment": p.get("price_value_assessment", {}).get("assessment", "No assessment available")
            }
            for p in underpriced_products[:5]  # Top 5 underpriced products
        ]
        
        # Construct the prompt for the LLM
        prompt = f"""
        Generate insights and recommendations based on the analysis of underpriced products in the enterprise networking manufacturing space.
        
        Overall Statistics:
        - Total underpriced products: {len(underpriced_products)}
        - Average underpriced percentage: {average_underpriced_pct:.2f}%
        
        Top Underpriced Products:
        {json.dumps(top_products, indent=2)}
        
        Category Summaries:
        {json.dumps(category_summaries, indent=2)}
        
        Manufacturer Summaries:
        {json.dumps(manufacturer_summaries, indent=2)}
        
        Please provide:
        1. Key insights about underpriced products in the enterprise networking manufacturing space
        2. Patterns or trends across categories and manufacturers
        3. Potential reasons for underpricing (e.g., market positioning, competitive strategy)
        4. Recommendations for buyers and manufacturers
        
        Format your response as a comprehensive analysis with clear sections and bullet points where appropriate.
        """
        
        # Call the LLM to generate insights
        response = await self.llm.ainvoke(prompt)
        
        # Extract the insights from the response
        insights = response.content.strip()
        
        return insights
    
    async def analyze_products(self, category: Optional[str] = None, manufacturer: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze products to identify underpriced ones in the enterprise networking manufacturing space.
        
        This is the main entry point for the product innovation agent. It runs the entire workflow
        from data collection to report generation.
        
        Args:
            category: Optional category to focus on (if None, all categories are analyzed)
            manufacturer: Optional manufacturer to focus on (if None, all manufacturers are analyzed)
            
        Returns:
            Dictionary with the analysis report
        """
        logger.info(f"Starting product innovation analysis for category={category}, manufacturer={manufacturer}")
        
        # Initialize the state with input parameters
        initial_state = {
            "category": category,
            "manufacturer": manufacturer
        }
        
        # Run the workflow
        try:
            # Execute the graph
            final_state = await self.graph.arun(initial_state)
            
            # Extract the report from the final state
            report = final_state.get("report", {})
            
            logger.info("Product innovation analysis completed successfully")
            
            return report
        except Exception as e:
            logger.error(f"Error in product innovation analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

async def main():
    """Main entry point for the product innovation agent."""
    try:
        # Initialize the agent
        agent = ProductInnovationAgent()
        
        # Get optional category and manufacturer from command line arguments
        import sys
        category = sys.argv[1] if len(sys.argv) > 1 else None
        manufacturer = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Run the analysis
        report = await agent.analyze_products(category, manufacturer)
        
        # Print the report summary
        print("\n=== Product Innovation Analysis Report ===\n")
        
        if "error" in report:
            print(f"Error: {report['error']}")
        else:
            summary = report.get("summary", {})
            print(f"Total products analyzed: {summary.get('total_products_analyzed', 0)}")
            print(f"Underpriced products found: {summary.get('underpriced_products_count', 0)} ({summary.get('underpriced_products_percentage', 0):.2f}%)")
            print(f"Average underpriced percentage: {summary.get('average_underpriced_percentage', 0):.2f}%\n")
            
            print("Top Underpriced Products:")
            for i, product in enumerate(report.get("top_underpriced_products", [])[:5], 1):
                print(f"{i}. {product.get('manufacturer', 'Unknown')} {product.get('name', 'Unknown')} {product.get('model', 'Unknown')}")
                print(f"   Category: {product.get('category', 'Unknown')}")
                print(f"   Underpriced by: {product.get('underpriced_percentage', 0):.2f}%")
                print(f"   Assessment: {product.get('assessment', 'No assessment available')[:100]}...\n")
            
            print("Key Insights:")
            insights = report.get("insights", "No insights available")
            # Print the first 500 characters of insights with ellipsis
            print(f"{insights[:500]}...\n")
            
            # Save the full report to a JSON file
            report_filename = f"product_innovation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"Full report saved to: {report_filename}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
