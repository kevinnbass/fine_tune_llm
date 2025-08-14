"""Input sanitization and security utilities."""

import re
import html
import urllib.parse
from typing import List, Optional, Dict, Any
import bleach
from bleach.css_sanitizer import CSSSanitizer
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize user inputs for security."""
    
    def __init__(self):
        """Initialize sanitizer with security rules."""
        # Allowed HTML tags (very restrictive for text classification)
        self.allowed_tags = []
        
        # Allowed attributes (none for text classification)
        self.allowed_attributes = {}
        
        # CSS sanitizer
        self.css_sanitizer = CSSSanitizer(allowed_css_properties=[])
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            # Script injection
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            
            # SQL injection patterns
            r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
            
            # XSS patterns
            r'on\w+\s*=',
            r'expression\s*\(',
            r'url\s*\(',
            
            # Path traversal
            r'\.\.[\\/]',
            r'[\\/]etc[\\/]',
            r'[\\/]proc[\\/]',
            
            # Command injection
            r'[;&|`$(){}\[\]<>]',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.dangerous_patterns]
    
    def sanitize(self, text: str, aggressive: bool = False) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Input text to sanitize
            aggressive: Whether to apply aggressive sanitization
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Basic length check
        if len(text) > 50000:  # Prevent DoS via large inputs
            text = text[:50000]
            logger.warning("Truncated oversized input")
        
        # HTML escape
        text = html.escape(text)
        
        # URL decode (in case of encoded attacks)
        text = urllib.parse.unquote(text)
        
        # Remove HTML/XML tags using bleach
        text = bleach.clean(
            text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            css_sanitizer=self.css_sanitizer,
            strip=True
        )
        
        if aggressive:
            # More aggressive sanitization
            text = self._aggressive_sanitize(text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _aggressive_sanitize(self, text: str) -> str:
        """Apply aggressive sanitization."""
        # Remove control characters except newline, tab, carriage return
        text = ''.join(char for char in text 
                      if ord(char) >= 32 or char in '\n\t\r')
        
        # Remove excessive punctuation
        text = re.sub(r'[!@#$%^&*()_+=\[\]{}|;:"<>?]{3,}', '', text)
        
        # Remove suspicious Unicode ranges
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
        
        return text
    
    def detect_threats(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect potential security threats in input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected threats
        """
        threats = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text)
            if matches:
                threat = {
                    'pattern_id': i,
                    'pattern': self.dangerous_patterns[i],
                    'matches': matches,
                    'risk_level': self._assess_risk_level(i, matches)
                }
                threats.append(threat)
        
        # Check for suspicious encoding
        if self._has_suspicious_encoding(text):
            threats.append({
                'pattern_id': -1,
                'pattern': 'suspicious_encoding',
                'matches': [],
                'risk_level': 'medium'
            })
        
        return threats
    
    def _assess_risk_level(self, pattern_id: int, matches: List[str]) -> str:
        """Assess risk level of detected pattern."""
        # High risk patterns
        high_risk_patterns = [0, 1, 2, 3, 4, 5, 6]  # Script, SQL, XSS
        
        if pattern_id in high_risk_patterns:
            return 'high'
        elif len(matches) > 3:
            return 'high'
        elif len(matches) > 1:
            return 'medium'
        else:
            return 'low'
    
    def _has_suspicious_encoding(self, text: str) -> bool:
        """Check for suspicious encoding patterns."""
        # Look for encoded characters that might bypass filters
        suspicious_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#x?[0-9a-fA-F]+;',  # HTML entities
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                encoded_chars = len(re.findall(pattern, text))
                if encoded_chars > 5:  # Threshold for suspicion
                    return True
        
        return False
    
    def is_safe(self, text: str, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Check if input is safe.
        
        Args:
            text: Input text
            strict: Whether to use strict checking
            
        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []
        
        # Check length
        if len(text) > 10000:
            issues.append("Input too long")
        
        # Detect threats
        threats = self.detect_threats(text)
        
        if threats:
            high_risk_threats = [t for t in threats if t['risk_level'] == 'high']
            medium_risk_threats = [t for t in threats if t['risk_level'] == 'medium']
            
            if high_risk_threats:
                issues.append(f"High risk threats detected: {len(high_risk_threats)}")
            
            if strict and medium_risk_threats:
                issues.append(f"Medium risk threats detected: {len(medium_risk_threats)}")
        
        # Check for binary content
        if self._has_binary_content(text):
            issues.append("Binary content detected")
        
        # Check for excessive repetition (potential DoS)
        if self._has_excessive_repetition(text):
            issues.append("Excessive repetition detected")
        
        return len(issues) == 0, issues
    
    def _has_binary_content(self, text: str) -> bool:
        """Check for binary content."""
        # Look for null bytes or excessive non-printable characters
        null_bytes = text.count('\x00')
        non_printable = sum(1 for char in text if ord(char) < 32 and char not in '\n\t\r')
        
        return null_bytes > 0 or (non_printable / len(text) > 0.1 if text else False)
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        if len(text) < 100:
            return False
        
        # Check for repeated characters
        max_repeat = 0
        current_char = None
        current_count = 0
        
        for char in text:
            if char == current_char:
                current_count += 1
                max_repeat = max(max_repeat, current_count)
            else:
                current_char = char
                current_count = 1
        
        return max_repeat > 50  # More than 50 repeated characters


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_requests_per_minute: int = 100,
        max_requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter.
        
        Args:
            redis_host: Redis host for storing rate limit data
            redis_port: Redis port
            max_requests_per_minute: Maximum requests per minute
            max_requests_per_hour: Maximum requests per hour
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        
        # Initialize Redis connection
        import redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
    
    async def check_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if within limits, False otherwise
        """
        import time
        
        current_time = int(time.time())
        minute_key = f"rate_limit:minute:{client_id}:{current_time // 60}"
        hour_key = f"rate_limit:hour:{client_id}:{current_time // 3600}"
        
        try:
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Increment counters
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            
            results = pipe.execute()
            
            minute_count = results[0]
            hour_count = results[2]
            
            # Check limits
            if minute_count > self.max_requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_id}: {minute_count}/min")
                return False
            
            if hour_count > self.max_requests_per_hour:
                logger.warning(f"Hourly rate limit exceeded for {client_id}: {hour_count}/hour")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiting fails
            return True
    
    def get_remaining_requests(self, client_id: str) -> Dict[str, int]:
        """
        Get remaining requests for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with remaining requests
        """
        import time
        
        current_time = int(time.time())
        minute_key = f"rate_limit:minute:{client_id}:{current_time // 60}"
        hour_key = f"rate_limit:hour:{client_id}:{current_time // 3600}"
        
        try:
            minute_count = int(self.redis_client.get(minute_key) or 0)
            hour_count = int(self.redis_client.get(hour_key) or 0)
            
            return {
                'remaining_per_minute': max(0, self.max_requests_per_minute - minute_count),
                'remaining_per_hour': max(0, self.max_requests_per_hour - hour_count)
            }
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return {
                'remaining_per_minute': self.max_requests_per_minute,
                'remaining_per_hour': self.max_requests_per_hour
            }