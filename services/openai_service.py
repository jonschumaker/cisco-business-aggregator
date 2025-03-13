#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI Service

This module provides a service for interacting with the OpenAI API.
It handles LLM requests, response processing, and error handling.

Key features:
- Support for both OpenAI and Azure OpenAI endpoints
- Asynchronous API calls
- Error handling and retries
- Caching integration
"""

import os
import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple

# Import local modules
from config import settings
from utils.cache import cache_llm_call, async_cache_with_ttl

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Service for interacting with the OpenAI API.
    
    This class provides methods for making requests to the OpenAI API,
    processing responses, and handling errors.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 azure_deployment_name: Optional[str] = None,
                 model: str = "gpt-4o",
                 provider: str = "azure_openai"):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: Optional API key for OpenAI. If not provided, it will be loaded from settings.
            azure_api_key: Optional API key for Azure OpenAI. If not provided, it will be loaded from settings.
            azure_endpoint: Optional endpoint for Azure OpenAI. If not provided, it will be loaded from settings.
            azure_deployment_name: Optional deployment name for Azure OpenAI. If not provided, it will be loaded from settings.
            model: The model to use for requests.
            provider: The provider to use (openai or azure_openai).
        """
        self.model = model
        self.provider = provider
        
        # Set API keys and endpoints
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.azure_api_key = azure_api_key or settings.AZURE_OPENAI_API_KEY
        self.azure_endpoint = azure_endpoint or settings.AZURE_OPENAI_ENDPOINT
        self.azure_deployment_name = azure_deployment_name or settings.AZURE_DEPLOYMENT_NAME
        
        # Validate API keys
        if self.provider == "openai" and not self.api_key:
            logger.error("OpenAI API key not provided and not found in settings")
            raise ValueError("OpenAI API key is required for OpenAI provider")
        
        if self.provider == "azure_openai" and not self.azure_api_key:
            logger.error("Azure OpenAI API key not provided and not found in settings")
            raise ValueError("Azure OpenAI API key is required for Azure OpenAI provider")
        
        # Initialize the OpenAI client
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized successfully with model {self.model}")
            else:  # azure_openai
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=self.azure_endpoint
                )
                logger.info(f"Azure OpenAI client initialized successfully with model {self.model}")
        except ImportError:
            logger.error("OpenAI client library not installed. Install with: pip install openai")
            raise ImportError("OpenAI client library not installed")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    @cache_llm_call
    async def chat_completion(self, 
                             messages: List[Dict[str, str]], 
                             temperature: float = 0.1,
                             max_tokens: Optional[int] = None,
                             top_p: float = 1.0,
                             frequency_penalty: float = 0.0,
                             presence_penalty: float = 0.0,
                             stop: Optional[Union[str, List[str]]] = None,
                             tools: Optional[List[Dict[str, Any]]] = None,
                             tool_choice: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a chat completion using the OpenAI API.
        
        This method is decorated with cache_llm_call to cache results.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Stop sequences
            tools: List of tools to use
            tool_choice: Tool choice parameter
            
        Returns:
            Dict[str, Any]: The chat completion response
        """
        logger.info(f"Generating chat completion with {len(messages)} messages")
        
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model if self.provider == "openai" else self.azure_deployment_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            # Add optional parameters if provided
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if stop:
                request_params["stop"] = stop
            
            if tools:
                request_params["tools"] = tools
            
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            # Execute the request
            # Note: We're using the synchronous method here, but wrapping it in asyncio.to_thread
            # to make it asynchronous.
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **request_params
            )
            
            # Convert the response to a dictionary
            response_dict = self._response_to_dict(response)
            
            logger.info(f"Chat completion generated successfully")
            return response_dict
        
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise
    
    @async_cache_with_ttl(settings.CACHE_TTL_LLM)
    async def chat_completion_with_retry(self, 
                                        messages: List[Dict[str, str]], 
                                        temperature: float = 0.1,
                                        max_tokens: Optional[int] = None,
                                        top_p: float = 1.0,
                                        frequency_penalty: float = 0.0,
                                        presence_penalty: float = 0.0,
                                        stop: Optional[Union[str, List[str]]] = None,
                                        tools: Optional[List[Dict[str, Any]]] = None,
                                        tool_choice: Optional[Union[str, Dict[str, str]]] = None,
                                        max_retries: int = 3,
                                        retry_delay: int = 5) -> Dict[str, Any]:
        """
        Generate a chat completion using the OpenAI API with retry logic.
        
        This method adds retry logic to the chat_completion method to handle temporary failures.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Stop sequences
            tools: List of tools to use
            tool_choice: Tool choice parameter
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: The chat completion response
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return await self.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice
                )
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Check if we should retry
                if retry_count < max_retries:
                    # Calculate delay with exponential backoff
                    delay = retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"Chat completion failed: {str(e)}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Chat completion failed after {max_retries} attempts: {str(e)}")
        
        # If we get here, all retries failed
        raise last_error or Exception("Chat completion failed after all retries")
    
    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert an OpenAI response object to a dictionary.
        
        Args:
            response: The OpenAI response object
            
        Returns:
            Dict[str, Any]: The response as a dictionary
        """
        # Check if the response is already a dictionary
        if isinstance(response, dict):
            return response
        
        # Try to convert the response to a dictionary
        try:
            # Convert to JSON string and then to dictionary
            return json.loads(json.dumps(response, default=lambda o: o.__dict__))
        except:
            # Fallback: extract the content from the response
            try:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": response.choices[0].message.content,
                                "role": response.choices[0].message.role
                            },
                            "finish_reason": response.choices[0].finish_reason
                        }
                    ],
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except:
                # Last resort: return a simple dictionary with the content
                try:
                    return {
                        "content": response.choices[0].message.content
                    }
                except:
                    logger.error("Failed to convert response to dictionary")
                    return {"error": "Failed to convert response to dictionary"}
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the content from a chat completion response.
        
        Args:
            response: The chat completion response
            
        Returns:
            str: The content of the response
        """
        try:
            # Try to extract content from the response
            if "choices" in response and len(response["choices"]) > 0:
                if "message" in response["choices"][0]:
                    if "content" in response["choices"][0]["message"]:
                        return response["choices"][0]["message"]["content"]
            
            # If we get here, we couldn't extract the content using the standard path
            # Try alternative paths
            if "content" in response:
                return response["content"]
            
            # Last resort: convert the response to a string
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting content from response: {str(e)}")
            return ""
    
    def extract_json_from_content(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON from content.
        
        Args:
            content: The content to extract JSON from
            
        Returns:
            Dict[str, Any]: The extracted JSON
        """
        try:
            # Try to extract JSON from the content
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to find object directly
                json_str = re.search(r'\{\s*".*"\s*:.*\}', content, re.DOTALL)
                if json_str:
                    json_str = json_str.group(0)
                else:
                    # If still no match, use the entire content
                    json_str = content
            
            # Parse the JSON string
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error extracting JSON from content: {str(e)}")
            return {}
