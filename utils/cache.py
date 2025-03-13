#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cache Utilities

This module provides caching functionality for the application.
It includes decorators for caching function results and utilities for managing the cache.

Key features:
- TTL-based caching
- Asynchronous caching
- Cache invalidation
- Cache statistics
"""

import os
import json
import logging
import hashlib
import functools
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Awaitable
from datetime import datetime, timedelta
import pickle
import time

# Import local modules
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Cache storage
_cache = {}
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "size": 0
}

def cache_with_ttl(ttl_seconds: int = 3600) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator for caching function results with a time-to-live (TTL).
    
    Args:
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R:
            # Check if caching is enabled
            if not settings.CACHE_ENABLED:
                return func(*args, **kwargs)
            
            # Generate a cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Check if the result is in the cache
            if cache_key in _cache:
                entry = _cache[cache_key]
                # Check if the entry is still valid
                if entry["expiry"] > time.time():
                    # Update cache stats
                    _cache_stats["hits"] += 1
                    logger.debug(f"Cache hit for {func.__name__}")
                    return entry["result"]
            
            # If not in cache or expired, call the function
            result = func(*args, **kwargs)
            
            # Store the result in the cache
            _cache[cache_key] = {
                "result": result,
                "expiry": time.time() + ttl_seconds
            }
            
            # Update cache stats
            _cache_stats["misses"] += 1
            _cache_stats["size"] = len(_cache)
            logger.debug(f"Cache miss for {func.__name__}")
            
            return result
        return wrapper
    return decorator

async def async_cache_with_ttl(ttl_seconds: int = 3600) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """
    Decorator for caching asynchronous function results with a time-to-live (TTL).
    
    Args:
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            # Check if caching is enabled
            if not settings.CACHE_ENABLED:
                return await func(*args, **kwargs)
            
            # Generate a cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Check if the result is in the cache
            if cache_key in _cache:
                entry = _cache[cache_key]
                # Check if the entry is still valid
                if entry["expiry"] > time.time():
                    # Update cache stats
                    _cache_stats["hits"] += 1
                    logger.debug(f"Cache hit for {func.__name__}")
                    return entry["result"]
            
            # If not in cache or expired, call the function
            result = await func(*args, **kwargs)
            
            # Store the result in the cache
            _cache[cache_key] = {
                "result": result,
                "expiry": time.time() + ttl_seconds
            }
            
            # Update cache stats
            _cache_stats["misses"] += 1
            _cache_stats["size"] = len(_cache)
            logger.debug(f"Cache miss for {func.__name__}")
            
            return result
        return wrapper
    return decorator

def cache_tavily_search(func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """
    Decorator for caching Tavily search results.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        # Check if caching is enabled
        if not settings.CACHE_ENABLED:
            return await func(*args, **kwargs)
        
        # Extract the query from the arguments
        query = kwargs.get("query", args[1] if len(args) > 1 else None)
        if not query:
            # If no query is provided, don't cache
            return await func(*args, **kwargs)
        
        # Generate a cache key
        cache_key = f"tavily_search_{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check if the result is in the cache
        if cache_key in _cache:
            entry = _cache[cache_key]
            # Check if the entry is still valid
            if entry["expiry"] > time.time():
                # Update cache stats
                _cache_stats["hits"] += 1
                logger.debug(f"Cache hit for Tavily search: {query}")
                return entry["result"]
        
        # If not in cache or expired, call the function
        result = await func(*args, **kwargs)
        
        # Store the result in the cache
        _cache[cache_key] = {
            "result": result,
            "expiry": time.time() + settings.CACHE_TTL_SEARCH
        }
        
        # Update cache stats
        _cache_stats["misses"] += 1
        _cache_stats["size"] = len(_cache)
        logger.debug(f"Cache miss for Tavily search: {query}")
        
        return result
    return wrapper

def cache_llm_call(func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """
    Decorator for caching LLM call results.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        # Check if caching is enabled
        if not settings.CACHE_ENABLED:
            return await func(*args, **kwargs)
        
        # Extract the messages from the arguments
        messages = kwargs.get("messages", args[1] if len(args) > 1 else None)
        if not messages:
            # If no messages are provided, don't cache
            return await func(*args, **kwargs)
        
        # Generate a cache key
        cache_key = f"llm_call_{hashlib.md5(str(messages).encode()).hexdigest()}"
        
        # Check if the result is in the cache
        if cache_key in _cache:
            entry = _cache[cache_key]
            # Check if the entry is still valid
            if entry["expiry"] > time.time():
                # Update cache stats
                _cache_stats["hits"] += 1
                logger.debug(f"Cache hit for LLM call")
                return entry["result"]
        
        # If not in cache or expired, call the function
        result = await func(*args, **kwargs)
        
        # Store the result in the cache
        _cache[cache_key] = {
            "result": result,
            "expiry": time.time() + settings.CACHE_TTL_LLM
        }
        
        # Update cache stats
        _cache_stats["misses"] += 1
        _cache_stats["size"] = len(_cache)
        logger.debug(f"Cache miss for LLM call")
        
        return result
    return wrapper

def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Generate a cache key for a function call.
    
    Args:
        func_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        str: Cache key
    """
    # Convert args and kwargs to a string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # Generate a hash of the function name and arguments
    key = f"{func_name}_{hashlib.md5((args_str + kwargs_str).encode()).hexdigest()}"
    
    return key

def clear_cache() -> None:
    """
    Clear the cache.
    """
    global _cache
    _cache = {}
    _cache_stats["size"] = 0
    logger.info("Cache cleared")

def invalidate_cache_entry(key: str) -> bool:
    """
    Invalidate a specific cache entry.
    
    Args:
        key: Cache key to invalidate
        
    Returns:
        bool: True if the entry was invalidated, False otherwise
    """
    if key in _cache:
        del _cache[key]
        _cache_stats["size"] = len(_cache)
        logger.info(f"Cache entry invalidated: {key}")
        return True
    return False

def get_cache_stats() -> Dict[str, int]:
    """
    Get cache statistics.
    
    Returns:
        Dict[str, int]: Cache statistics
    """
    return _cache_stats.copy()

def save_cache_to_disk(file_path: Optional[str] = None) -> str:
    """
    Save the cache to disk.
    
    Args:
        file_path: Path to save the cache to. If not provided, a default path will be used.
        
    Returns:
        str: Path to the saved cache file
    """
    if file_path is None:
        file_path = os.path.join(settings.CACHE_DIRECTORY, "cache.pkl")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the cache to disk
    with open(file_path, 'wb') as f:
        pickle.dump(_cache, f)
    
    logger.info(f"Cache saved to disk: {file_path}")
    
    return file_path

def load_cache_from_disk(file_path: Optional[str] = None) -> bool:
    """
    Load the cache from disk.
    
    Args:
        file_path: Path to load the cache from. If not provided, a default path will be used.
        
    Returns:
        bool: True if the cache was loaded successfully, False otherwise
    """
    global _cache
    
    if file_path is None:
        file_path = os.path.join(settings.CACHE_DIRECTORY, "cache.pkl")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.warning(f"Cache file not found: {file_path}")
        return False
    
    try:
        # Load the cache from disk
        with open(file_path, 'rb') as f:
            _cache = pickle.load(f)
        
        # Update cache stats
        _cache_stats["size"] = len(_cache)
        
        logger.info(f"Cache loaded from disk: {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading cache from disk: {str(e)}")
        return False

# Initialize the cache
def initialize_cache() -> None:
    """
    Initialize the cache.
    """
    # Create the cache directory if it doesn't exist
    os.makedirs(settings.CACHE_DIRECTORY, exist_ok=True)
    
    # Try to load the cache from disk
    cache_file = os.path.join(settings.CACHE_DIRECTORY, "cache.pkl")
    if os.path.exists(cache_file):
        load_cache_from_disk(cache_file)
    
    logger.info("Cache initialized")

# Initialize the cache when the module is imported
initialize_cache()
