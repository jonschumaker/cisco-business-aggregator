#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Settings

This module provides centralized configuration settings for the application.
It loads settings from environment variables and provides default values
for various configuration options.

Key features:
- Environment variable loading
- Default configuration values
- Configuration validation
- Settings access functions
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== Logging Configuration =====

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== API Keys and Credentials =====

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://phx-sales-ai.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Set Azure OpenAI environment variables with the correct names
# This resolves the environment variable name mismatch issue
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "https://phx-sales-ai.openai.azure.com/")
os.environ["AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Tavily API settings
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ===== Storage Configuration =====

# Google Cloud Storage settings
CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "secrets", 
    "google-credentials-dev.json"
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

GCS_BUCKET_PATH = os.getenv("OUTCOMES_PATH", "gs://sales-ai-dev-outcomes-6f1ce1c")
GCS_BUCKET_NAME = GCS_BUCKET_PATH.replace("gs://", "").split("/")[0]
GCS_FOLDER = "news-reports"
GCS_EXCEL_FOLDER = "data"  # Folder where Excel files are stored in GCS

# Local storage settings
LOCAL_REPORTS_DIR = os.getenv("LOCAL_REPORTS_DIR", "reports")
USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "true").lower() in ["true", "1", "yes"]
USE_GCS_EXCEL = True  # Always use GCS for Excel files

# ===== Cache Configuration =====

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ["true", "1", "yes"]
CACHE_DIRECTORY = os.getenv("CACHE_DIRECTORY", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))
CACHE_TTL_SEARCH = int(os.getenv("CACHE_TTL_SEARCH", 86400))  # 1 day in seconds
CACHE_TTL_LLM = int(os.getenv("CACHE_TTL_LLM", 604800))  # 1 week in seconds

# ===== Application Configuration =====

# Product categories for product innovation agent
PRODUCT_CATEGORIES = [
    "enterprise computer hardware",
    "cybersecurity appliances",
    "firewall systems",
    "networking equipment",
    "network switches",
    "cyber defense solutions",
    "enterprise computer parts"
]

# Target manufacturers for product innovation agent
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

# Criteria for determining if a product is underpriced
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

# ===== Validation and Initialization =====

def validate_settings() -> bool:
    """
    Validate that all required settings are available.
    
    Returns:
        bool: True if all required settings are available, False otherwise
    """
    missing_settings = []
    
    # Check API keys
    if not TAVILY_API_KEY:
        missing_settings.append("TAVILY_API_KEY")
    
    if not OPENAI_API_KEY and not AZURE_OPENAI_API_KEY:
        missing_settings.append("OPENAI_API_KEY or AZURE_OPENAI_API_KEY")
    
    # Log missing settings
    if missing_settings:
        logger.error(f"Missing required settings: {', '.join(missing_settings)}")
        return False
    
    return True

def initialize_settings() -> None:
    """
    Initialize settings and create necessary directories.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    
    # Create reports directory if it doesn't exist and using local storage
    if USE_LOCAL_STORAGE:
        os.makedirs(LOCAL_REPORTS_DIR, exist_ok=True)
    
    # Create secrets directory if it doesn't exist
    secrets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "secrets")
    os.makedirs(secrets_dir, exist_ok=True)
    
    # Validate settings
    if not validate_settings():
        logger.warning("Some required settings are missing. The application may not function correctly.")

def get_settings() -> Dict[str, Any]:
    """
    Get all settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary of all settings
    """
    return {
        "openai_api_key": OPENAI_API_KEY,
        "azure_openai_api_key": AZURE_OPENAI_API_KEY,
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_openai_api_version": AZURE_OPENAI_API_VERSION,
        "azure_deployment_name": AZURE_DEPLOYMENT_NAME,
        "tavily_api_key": TAVILY_API_KEY,
        "gcs_bucket_path": GCS_BUCKET_PATH,
        "gcs_bucket_name": GCS_BUCKET_NAME,
        "gcs_folder": GCS_FOLDER,
        "gcs_excel_folder": GCS_EXCEL_FOLDER,
        "local_reports_dir": LOCAL_REPORTS_DIR,
        "use_local_storage": USE_LOCAL_STORAGE,
        "use_gcs_excel": USE_GCS_EXCEL,
        "cache_enabled": CACHE_ENABLED,
        "cache_directory": CACHE_DIRECTORY,
        "cache_ttl_search": CACHE_TTL_SEARCH,
        "cache_ttl_llm": CACHE_TTL_LLM,
        "product_categories": PRODUCT_CATEGORIES,
        "target_manufacturers": TARGET_MANUFACTURERS,
        "underpriced_criteria": UNDERPRICED_CRITERIA
    }

# Initialize settings when the module is imported
initialize_settings()
