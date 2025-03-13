#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Package

This package provides centralized configuration for the application.
It includes settings, logging configuration, and other configuration-related
functionality.

Usage:
    from config import settings
    from config.logging_config import get_logger
    
    logger = get_logger(__name__)
    api_key = settings.OPENAI_API_KEY
"""

from config import settings
from config import logging_config

# Export key functions and variables for easy access
get_logger = logging_config.get_logger
get_settings = settings.get_settings
validate_settings = settings.validate_settings
