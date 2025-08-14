"""
API adapter for hexagonal architecture.

This module provides HTTP API operations through the adapter pattern,
supporting external service communication with resilience patterns.
"""

import requests
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import logging

from ..core.interfaces import APIPort
from ..core.exceptions import IntegrationError, NetworkError
from ..utils.resilience import circuit_breaker, retry

logger = logging.getLogger(__name__)


class APIAdapter(APIPort):
    """
    API adapter implementing HTTP client operations.
    
    Provides HTTP communication with external services, including
    authentication, request/response handling, and resilience patterns.
    """
    
    def __init__(self, 
                 base_url: str,
                 timeout: int = 30,
                 max_retries: int = 3,
                 enable_resilience: bool = True,
                 api_key: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None):
        """
        Initialize API adapter.
        
        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_resilience: Enable circuit breaker and retry patterns
            api_key: API key for authentication
            headers: Default headers for requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_resilience = enable_resilience
        self.api_key = api_key
        
        # Default headers
        self.default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'FineTuneLLM-APIAdapter/2.0'
        }
        
        if headers:
            self.default_headers.update(headers)
            
        if self.api_key:
            self.default_headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        
        logger.info(f"Initialized APIAdapter with base_url: {self.base_url}")
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(self.base_url + '/', endpoint.lstrip('/'))
    
    def _update_stats(self, success: bool, response_time: float):
        """Update request statistics."""
        self.stats['total_requests'] += 1
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # Update average response time
        total = self.stats['total_requests']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = ((current_avg * (total - 1)) + response_time) / total
    
    @circuit_breaker("api_requests")
    @retry(max_attempts=3, base_delay=1.0)
    def get(self, 
            endpoint: str, 
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Response data as dictionary
            
        Raises:
            IntegrationError: If request fails
        """
        url = self._build_url(endpoint)
        request_headers = self.default_headers.copy()
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except ValueError:
                # If not JSON, return text content
                data = {'content': response.text, 'status_code': response.status_code}
            
            self._update_stats(True, response_time)
            
            logger.debug(f"GET {url} - Status: {response.status_code}, Time: {response_time:.3f}s")
            return data
            
        except requests.exceptions.RequestException as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"GET {url} failed: {e}")
            raise IntegrationError(f"GET request failed: {e}")
    
    @circuit_breaker("api_requests")
    @retry(max_attempts=3, base_delay=1.0)
    def post(self, 
             endpoint: str, 
             data: Optional[Dict[str, Any]] = None,
             json_data: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Response data as dictionary
        """
        url = self._build_url(endpoint)
        request_headers = self.default_headers.copy()
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            response = self.session.post(
                url,
                data=data,
                json=json_data,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except ValueError:
                result = {'content': response.text, 'status_code': response.status_code}
            
            self._update_stats(True, response_time)
            
            logger.debug(f"POST {url} - Status: {response.status_code}, Time: {response_time:.3f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"POST {url} failed: {e}")
            raise IntegrationError(f"POST request failed: {e}")
    
    @circuit_breaker("api_requests")
    @retry(max_attempts=3, base_delay=1.0)
    def put(self, 
            endpoint: str, 
            data: Optional[Dict[str, Any]] = None,
            json_data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make PUT request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Response data as dictionary
        """
        url = self._build_url(endpoint)
        request_headers = self.default_headers.copy()
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            response = self.session.put(
                url,
                data=data,
                json=json_data,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except ValueError:
                result = {'content': response.text, 'status_code': response.status_code}
            
            self._update_stats(True, response_time)
            
            logger.debug(f"PUT {url} - Status: {response.status_code}, Time: {response_time:.3f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"PUT {url} failed: {e}")
            raise IntegrationError(f"PUT request failed: {e}")
    
    @circuit_breaker("api_requests")
    @retry(max_attempts=3, base_delay=1.0)
    def delete(self, 
               endpoint: str, 
               params: Optional[Dict[str, Any]] = None,
               headers: Optional[Dict[str, str]] = None,
               timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Make DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            Response data as dictionary
        """
        url = self._build_url(endpoint)
        request_headers = self.default_headers.copy()
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            response = self.session.delete(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except ValueError:
                result = {'content': response.text, 'status_code': response.status_code}
            
            self._update_stats(True, response_time)
            
            logger.debug(f"DELETE {url} - Status: {response.status_code}, Time: {response_time:.3f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"DELETE {url} failed: {e}")
            raise IntegrationError(f"DELETE request failed: {e}")
    
    def upload_file(self, 
                    endpoint: str,
                    file_path: str,
                    file_field: str = 'file',
                    additional_data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload file to API endpoint.
        
        Args:
            endpoint: API endpoint for file upload
            file_path: Path to file to upload
            file_field: Form field name for file
            additional_data: Additional form data
            headers: Additional headers
            
        Returns:
            Response data as dictionary
        """
        url = self._build_url(endpoint)
        request_headers = {}
        
        # Don't set Content-Type for file uploads (let requests handle it)
        for key, value in self.default_headers.items():
            if key.lower() != 'content-type':
                request_headers[key] = value
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            with open(file_path, 'rb') as f:
                files = {file_field: f}
                data = additional_data or {}
                
                response = self.session.post(
                    url,
                    files=files,
                    data=data,
                    headers=request_headers,
                    timeout=self.timeout
                )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except ValueError:
                result = {'content': response.text, 'status_code': response.status_code}
            
            self._update_stats(True, response_time)
            
            logger.debug(f"FILE UPLOAD {url} - Status: {response.status_code}, Time: {response_time:.3f}s")
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"File upload to {url} failed: {e}")
            raise IntegrationError(f"File upload failed: {e}")
    
    def download_file(self, 
                     endpoint: str, 
                     local_path: str,
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Download file from API endpoint.
        
        Args:
            endpoint: API endpoint for file download
            local_path: Local path to save file
            params: Query parameters
            headers: Additional headers
            
        Returns:
            True if successful
        """
        url = self._build_url(endpoint)
        request_headers = self.default_headers.copy()
        
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.now()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=request_headers,
                timeout=self.timeout,
                stream=True
            )
            
            response.raise_for_status()
            
            # Write file in chunks
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(True, response_time)
            
            logger.debug(f"FILE DOWNLOAD {url} -> {local_path} - Time: {response_time:.3f}s")
            return True
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, response_time)
            
            logger.error(f"File download from {url} failed: {e}")
            raise IntegrationError(f"File download failed: {e}")
    
    def health_check(self, endpoint: str = '/health') -> Dict[str, Any]:
        """
        Perform health check on API.
        
        Args:
            endpoint: Health check endpoint
            
        Returns:
            Health check results
        """
        try:
            start_time = datetime.now()
            response = self.get(endpoint, timeout=5)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'response': response
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get API adapter statistics.
        
        Returns:
            Dictionary with statistics
        """
        total = self.stats['total_requests']
        success_rate = (self.stats['successful_requests'] / total * 100) if total > 0 else 0
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'base_url': self.base_url,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }
    
    def reset_statistics(self):
        """Reset request statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        logger.info("API adapter statistics reset")
    
    def close(self):
        """Close the session."""
        try:
            self.session.close()
            logger.info("API adapter session closed")
        except Exception as e:
            logger.error(f"Error closing API adapter session: {e}")