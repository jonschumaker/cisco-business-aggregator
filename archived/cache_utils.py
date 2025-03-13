#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cache Utilities for Product Innovation Agent

This module provides caching functionality for API calls to improve performance
and reduce redundant API usage. It implements a disk-based caching system using
the diskcache library with configurable TTL (time-to-live) settings.

Key features:
- Disk-based caching for Tavily search results and LLM API calls
- Configurable TTL settings for different types of cached data
- Cache key generation based on query parameters
- Cache invalidation logic
"""

import os
import json
import hashlib
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
from functools import wraps
from diskcache import Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default cache settings
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_CACHE_TTL_SEARCH = 86400  # 1 day in seconds
DEFAULT_CACHE_TTL_LLM = 604800    # 1 week in seconds

# Load cache settings from environment variables
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ["true", "1", "yes"]
CACHE_DIRECTORY = os.getenv("CACHE_DIRECTORY", DEFAULT_CACHE_DIR)
CACHE_TTL_SEARCH = int(os.getenv("CACHE_TTL_SEARCH", DEFAULT_CACHE_TTL_SEARCH))
CACHE_TTL_LLM = int(os.getenv("CACHE_TTL_LLM", DEFAULT_CACHE_TTL_LLM))

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIRECTORY, exist_ok=True)

# Initialize the cache
cache = Cache(CACHE_DIRECTORY)

def generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate a cache key based on the prefix and kwargs.
    
    Args:
        prefix: A prefix for the cache key (e.g., 'tavily_search', 'llm_call')
        **kwargs: The parameters used for the API call
        
    Returns:
        A unique cache key as a string
    """
    # Convert kwargs to a sorted, stringified representation
    kwargs_str = json.dumps(kwargs, sort_keys=True)
    
    # Generate a hash of the kwargs string
    kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
    
    # Combine prefix and hash to create the cache key
    return f"{prefix}_{kwargs_hash}"

def cache_tavily_search(func):
    """
    Decorator to cache Tavily search results.
    
    Args:
        func: The function to decorate (should be a Tavily search function)
        
    Returns:
        The decorated function with caching
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not CACHE_ENABLED:
            return await func(*args, **kwargs)
        
        # Generate a cache key
        cache_key = generate_cache_key("tavily_search", **kwargs)
        
        # Check if the result is in the cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for Tavily search: {cache_key}")
            return cached_result
        
        # If not in cache, call the function
        result = await func(*args, **kwargs)
        
        # Store the result in the cache
        cache.set(cache_key, result, expire=CACHE_TTL_SEARCH)
        logger.info(f"Cached Tavily search result: {cache_key}")
        
        return result
    
    return wrapper

def cache_llm_call(func):
    """
    Decorator to cache LLM API calls.
    
    Args:
        func: The function to decorate (should be an LLM API call function)
        
    Returns:
        The decorated function with caching
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not CACHE_ENABLED:
            return await func(*args, **kwargs)
        
        # Extract the prompt from kwargs or args
        prompt = kwargs.get("prompt", None)
        if prompt is None and len(args) > 0:
            prompt = args[0]
        
        # If no prompt found, skip caching
        if prompt is None:
            return await func(*args, **kwargs)
        
        # Generate a cache key
        cache_key = generate_cache_key("llm_call", prompt=prompt, **kwargs)
        
        # Check if the result is in the cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for LLM call: {cache_key[:30]}...")
            return cached_result
        
        # If not in cache, call the function
        result = await func(*args, **kwargs)
        
        # Store the result in the cache
        cache.set(cache_key, result, expire=CACHE_TTL_LLM)
        logger.info(f"Cached LLM call result: {cache_key[:30]}...")
        
        return result
    
    return wrapper

def clear_cache(prefix: Optional[str] = None) -> int:
    """
    Clear the cache, optionally filtering by prefix.
    
    Args:
        prefix: Optional prefix to filter which cache entries to clear
        
    Returns:
        Number of cache entries cleared
    """
    if prefix:
        # Clear only entries with the specified prefix
        count = 0
        for key in list(cache):
            if str(key).startswith(prefix):
                cache.delete(key)
                count += 1
        logger.info(f"Cleared {count} cache entries with prefix '{prefix}'")
        return count
    else:
        # Clear the entire cache
        count = len(cache)
        cache.clear()
        logger.info(f"Cleared entire cache ({count} entries)")
        return count

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "cache_enabled": CACHE_ENABLED,
        "cache_directory": CACHE_DIRECTORY,
        "cache_size": len(cache),
        "cache_disk_size": cache.volume() if hasattr(cache, 'volume') else "Unknown",
        "tavily_search_ttl": CACHE_TTL_SEARCH,
        "llm_call_ttl": CACHE_TTL_LLM,
        "tavily_entries": 0,
        "llm_entries": 0,
        "other_entries": 0
    }
    
    # Count entries by type
    for key in cache:
        key_str = str(key)
        if key_str.startswith("tavily_search"):
            stats["tavily_entries"] += 1
        elif key_str.startswith("llm_call"):
            stats["llm_entries"] += 1
        else:
            stats["other_entries"] += 1
    
    return stats

def invalidate_cache_entry(cache_key: str) -> bool:
    """
    Invalidate a specific cache entry.
    
    Args:
        cache_key: The cache key to invalidate
        
    Returns:
        True if the entry was found and invalidated, False otherwise
    """
    if cache_key in cache:
        cache.delete(cache_key)
        logger.info(f"Invalidated cache entry: {cache_key}")
        return True
    else:
        logger.warning(f"Cache entry not found: {cache_key}")
        return False

def cache_with_ttl(ttl: int):
    """
    Decorator factory to cache function results with a specific TTL.
    
    Args:
        ttl: Time-to-live in seconds for the cache entry
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return func(*args, **kwargs)
            
            # Generate a cache key
            func_name = func.__name__
            cache_key = generate_cache_key(func_name, *args, **kwargs)
            
            # Check if the result is in the cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func_name}: {cache_key}")
                return cached_result
            
            # If not in cache, call the function
            result = func(*args, **kwargs)
            
            # Store the result in the cache
            cache.set(cache_key, result, expire=ttl)
            logger.info(f"Cached {func_name} result: {cache_key}")
            
            return result
        
        return wrapper
    
    return decorator

# Async version of cache_with_ttl
def async_cache_with_ttl(ttl: int):
    """
    Decorator factory to cache async function results with a specific TTL.
    
    Args:
        ttl: Time-to-live in seconds for the cache entry
        
    Returns:
        A decorator function for async functions
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return await func(*args, **kwargs)
            
            # Generate a cache key
            func_name = func.__name__
            cache_key = generate_cache_key(func_name, *args, **kwargs)
            
            # Check if the result is in the cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func_name}: {cache_key}")
                return cached_result
            
            # If not in cache, call the function
            result = await func(*args, **kwargs)
            
            # Store the result in the cache
            cache.set(cache_key, result, expire=ttl)
            logger.info(f"Cached {func_name} result: {cache_key}")
            
            return result
        
        return wrapper
    
    return decorator
