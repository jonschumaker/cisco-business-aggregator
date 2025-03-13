#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging Configuration

This module provides centralized logging configuration for the application.
It sets up logging handlers, formatters, and log levels for different components
of the application.

Key features:
- Configurable log levels
- File and console logging
- Component-specific logging
- Log rotation
"""

import os
import logging
import logging.handlers
from typing import Dict, Any, Optional

# ===== Logging Configuration =====

# Default log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Component-specific log levels
COMPONENT_LOG_LEVELS = {
    "agents.product_innovation": logging.INFO,
    "agents.research": logging.INFO,
    "agents.url_finder": logging.INFO,
    "utils.cache": logging.INFO,
    "utils.storage": logging.INFO,
    "utils.file_utils": logging.INFO,
    "utils.report_utils": logging.INFO,
    "interfaces": logging.INFO,
    "models": logging.INFO,
    "services": logging.INFO,
    "config": logging.INFO
}

def setup_logging(log_dir: str = LOG_DIR, log_level: int = DEFAULT_LOG_LEVEL) -> None:
    """
    Set up logging for the application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Default log level for the application
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Create file handler with rotation
    main_log_file = os.path.join(log_dir, "application.log")
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Set up component-specific loggers
    for component, level in COMPONENT_LOG_LEVELS.items():
        component_logger = logging.getLogger(component)
        component_logger.setLevel(level)
        
        # Create component-specific file handler with rotation
        component_log_file = os.path.join(log_dir, f"{component.replace('.', '_')}.log")
        component_handler = logging.handlers.RotatingFileHandler(
            component_log_file, 
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3
        )
        component_handler.setFormatter(formatter)
        component_handler.setLevel(level)
        component_logger.addHandler(component_handler)
    
    # Log that logging has been set up
    logging.info(f"Logging initialized. Log files will be stored in {log_dir}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Name of the component
        
    Returns:
        logging.Logger: Logger for the component
    """
    return logging.getLogger(name)

def set_log_level(name: str, level: int) -> None:
    """
    Set the log level for a specific component.
    
    Args:
        name: Name of the component
        level: Log level to set
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Update handlers
    for handler in logger.handlers:
        handler.setLevel(level)
    
    # Update component log level in the dictionary
    if name in COMPONENT_LOG_LEVELS:
        COMPONENT_LOG_LEVELS[name] = level

def get_log_levels() -> Dict[str, int]:
    """
    Get the current log levels for all components.
    
    Returns:
        Dict[str, int]: Dictionary of component names and log levels
    """
    return COMPONENT_LOG_LEVELS.copy()

# Initialize logging when the module is imported
setup_logging()
