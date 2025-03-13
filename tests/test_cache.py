#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the cache utilities.

This module contains tests for the cache utilities in utils/cache.py.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock

# Import the module to test
from utils.cache import (
    cache_with_ttl,
    clear_cache,
    get_cache_stats,
    _cache,
    _cache_stats,
    _generate_cache_key
)

# Test the cache_with_ttl decorator
def test_cache_with_ttl():
    """Test that the cache_with_ttl decorator caches function results."""
    # Create a mock function that returns a different value each time
    mock_func = MagicMock(side_effect=range(10))
    
    # Apply the cache decorator
    cached_func = cache_with_ttl(ttl_seconds=60)(mock_func)
    
    # Call the function multiple times with the same arguments
    result1 = cached_func(1, 2, c=3)
    result2 = cached_func(1, 2, c=3)
    
    # The function should only be called once, and both results should be the same
    assert mock_func.call_count == 1
    assert result1 == result2
    
    # Call the function with different arguments
    result3 = cached_func(4, 5, c=6)
    
    # The function should be called again, and the result should be different
    assert mock_func.call_count == 2
    assert result1 != result3

# Test cache expiration
def test_cache_expiration():
    """Test that cached results expire after the TTL."""
    # Create a mock function
    mock_func = MagicMock(side_effect=range(10))
    
    # Apply the cache decorator with a short TTL
    cached_func = cache_with_ttl(ttl_seconds=1)(mock_func)
    
    # Call the function
    result1 = cached_func(1, 2, c=3)
    
    # Wait for the cache to expire
    time.sleep(1.1)
    
    # Call the function again with the same arguments
    result2 = cached_func(1, 2, c=3)
    
    # The function should be called twice, and the results should be different
    assert mock_func.call_count == 2
    assert result1 != result2

# Test cache statistics
def test_cache_stats():
    """Test that cache statistics are updated correctly."""
    # Clear the cache and reset stats
    clear_cache()
    
    # Create a mock function
    mock_func = MagicMock(side_effect=range(10))
    
    # Apply the cache decorator
    cached_func = cache_with_ttl(ttl_seconds=60)(mock_func)
    
    # Initial stats
    initial_stats = get_cache_stats()
    assert initial_stats["hits"] == 0
    assert initial_stats["misses"] == 0
    assert initial_stats["size"] == 0
    
    # Call the function (should be a miss)
    cached_func(1, 2, c=3)
    
    # Stats after first call
    stats_after_miss = get_cache_stats()
    assert stats_after_miss["hits"] == 0
    assert stats_after_miss["misses"] == 1
    assert stats_after_miss["size"] == 1
    
    # Call the function again with the same arguments (should be a hit)
    cached_func(1, 2, c=3)
    
    # Stats after second call
    stats_after_hit = get_cache_stats()
    assert stats_after_hit["hits"] == 1
    assert stats_after_hit["misses"] == 1
    assert stats_after_hit["size"] == 1

# Test cache key generation
def test_generate_cache_key():
    """Test that cache keys are generated correctly."""
    # Generate keys for different function calls
    key1 = _generate_cache_key("func1", (1, 2), {"c": 3})
    key2 = _generate_cache_key("func1", (1, 2), {"c": 3})
    key3 = _generate_cache_key("func1", (1, 2), {"c": 4})
    key4 = _generate_cache_key("func2", (1, 2), {"c": 3})
    
    # Keys for the same function and arguments should be the same
    assert key1 == key2
    
    # Keys for different arguments or functions should be different
    assert key1 != key3
    assert key1 != key4

# Test clearing the cache
def test_clear_cache():
    """Test that the cache can be cleared."""
    # Create a mock function
    mock_func = MagicMock(side_effect=range(10))
    
    # Apply the cache decorator
    cached_func = cache_with_ttl(ttl_seconds=60)(mock_func)
    
    # Call the function to populate the cache
    cached_func(1, 2, c=3)
    
    # Cache should have one entry
    assert len(_cache) == 1
    
    # Clear the cache
    clear_cache()
    
    # Cache should be empty
    assert len(_cache) == 0
    
    # Stats should be reset
    stats = get_cache_stats()
    assert stats["size"] == 0

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
