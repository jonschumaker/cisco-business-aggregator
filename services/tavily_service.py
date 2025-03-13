#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tavily Service

This module provides a service for interacting with the Tavily search API.
It handles search requests, result processing, and error handling.

Key features:
- Asynchronous search requests
- Result processing and filtering
- Error handling and retries
- Caching integration
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Import local modules
from config import settings
from utils.cache import cache_tavily_search, async_cache_with_ttl

# Configure logging
logger = logging.getLogger(__name__)

class TavilyService:
    """
    Service for interacting with the Tavily search API.
    
    This class provides methods for searching the web using the Tavily API,
    processing search results, and handling errors.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily service.
        
        Args:
            api_key: Optional API key for Tavily. If not provided, it will be loaded from settings.
        """
        self.api_key = api_key or settings.TAVILY_API_KEY
        
        if not self.api_key:
            logger.error("Tavily API key not provided and not found in settings")
            raise ValueError("Tavily API key is required")
        
        # Initialize the Tavily client
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("Tavily client initialized successfully")
        except ImportError:
            logger.error("Tavily client library not installed. Install with: pip install tavily")
            raise ImportError("Tavily client library not installed")
        except Exception as e:
            logger.error(f"Error initializing Tavily client: {str(e)}")
            raise
    
    @cache_tavily_search
    async def search(self, query: str, search_depth: str = "basic", 
                    include_answer: bool = True, include_domains: Optional[List[str]] = None,
                    exclude_domains: Optional[List[str]] = None, 
                    max_results: int = 10, days_back: int = 365) -> Dict[str, Any]:
        """
        Search the web using the Tavily API.
        
        This method is decorated with cache_tavily_search to cache results.
        
        Args:
            query: The search query
            search_depth: The search depth (basic or advanced)
            include_answer: Whether to include an answer in the response
            include_domains: List of domains to include in the search
            exclude_domains: List of domains to exclude from the search
            max_results: Maximum number of results to return
            days_back: Number of days to look back for results
            
        Returns:
            Dict[str, Any]: The search results
        """
        logger.info(f"Performing Tavily search with query: '{query}'")
        
        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "max_results": max_results
            }
            
            # Add optional parameters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Add time filter if days_back is provided
            if days_back and days_back < 365:
                # Calculate the date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Format dates as YYYY-MM-DD
                search_params["start_published_date"] = start_date.strftime("%Y-%m-%d")
                search_params["end_published_date"] = end_date.strftime("%Y-%m-%d")
            
            # Execute the search
            # Note: We're using the synchronous method here, but wrapping it in asyncio.to_thread
            # to make it asynchronous. This is because the Tavily client doesn't have native async support.
            result = await asyncio.to_thread(
                self.client.search,
                **search_params
            )
            
            logger.info(f"Tavily search completed successfully. Found {len(result.get('results', []))} results.")
            return result
        
        except Exception as e:
            logger.error(f"Error performing Tavily search: {str(e)}")
            raise
    
    @async_cache_with_ttl(settings.CACHE_TTL_SEARCH)
    async def search_with_retry(self, query: str, search_depth: str = "basic", 
                               include_answer: bool = True, include_domains: Optional[List[str]] = None,
                               exclude_domains: Optional[List[str]] = None, 
                               max_results: int = 10, days_back: int = 365,
                               max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
        """
        Search the web using the Tavily API with retry logic.
        
        This method adds retry logic to the search method to handle temporary failures.
        
        Args:
            query: The search query
            search_depth: The search depth (basic or advanced)
            include_answer: Whether to include an answer in the response
            include_domains: List of domains to include in the search
            exclude_domains: List of domains to exclude from the search
            max_results: Maximum number of results to return
            days_back: Number of days to look back for results
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: The search results
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return await self.search(
                    query=query,
                    search_depth=search_depth,
                    include_answer=include_answer,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    max_results=max_results,
                    days_back=days_back
                )
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Check if we should retry
                if retry_count < max_retries:
                    # Calculate delay with exponential backoff
                    delay = retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Tavily search failed: {str(e)}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tavily search failed after {max_retries} attempts: {str(e)}")
        
        # If we get here, all retries failed
        raise last_error or Exception("Tavily search failed after all retries")
    
    def extract_urls_from_results(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract URLs from search results.
        
        Args:
            results: The search results from Tavily
            
        Returns:
            List[str]: List of URLs extracted from the results
        """
        urls = []
        
        # Extract URLs from the results
        if "results" in results:
            for result in results["results"]:
                if "url" in result:
                    urls.append(result["url"])
        
        # Extract URLs from the context
        if "context" in results:
            for context_item in results["context"]:
                if "url" in context_item:
                    urls.append(context_item["url"])
        
        # Remove duplicates
        urls = list(dict.fromkeys(urls))
        
        return urls
    
    def extract_content_from_results(self, results: Dict[str, Any]) -> str:
        """
        Extract content from search results.
        
        Args:
            results: The search results from Tavily
            
        Returns:
            str: Combined content from the results
        """
        content = ""
        
        # Extract answer if available
        if "answer" in results and results["answer"]:
            content += results["answer"] + "\n\n"
        
        # Extract content from the context
        if "context" in results:
            for context_item in results["context"]:
                if "content" in context_item and context_item["content"]:
                    content += context_item["content"] + "\n\n"
        
        return content.strip()
