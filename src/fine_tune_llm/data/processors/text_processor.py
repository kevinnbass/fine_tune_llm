"""
Text processing for LLM training data.

This module provides text preprocessing, cleaning, and formatting
capabilities for language model training datasets.
"""

import re
import html
import unicodedata
from typing import Dict, List, Optional, Any, Union
import logging

from .base import BaseDataProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseDataProcessor):
    """
    Text processor for LLM training data.
    
    Provides comprehensive text cleaning, normalization, and formatting
    capabilities for preparing high-quality training data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Text processing configuration
        self.text_config = self.config.get('text', {})
        
        # Cleaning options
        self.clean_html = self.text_config.get('clean_html', True)
        self.clean_urls = self.text_config.get('clean_urls', True)
        self.clean_emails = self.text_config.get('clean_emails', True)
        self.clean_phone_numbers = self.text_config.get('clean_phone_numbers', True)
        self.clean_extra_whitespace = self.text_config.get('clean_extra_whitespace', True)
        self.clean_special_chars = self.text_config.get('clean_special_chars', False)
        
        # Normalization options
        self.normalize_unicode = self.text_config.get('normalize_unicode', True)
        self.normalize_case = self.text_config.get('normalize_case', False)
        self.normalize_quotes = self.text_config.get('normalize_quotes', True)
        self.normalize_dashes = self.text_config.get('normalize_dashes', True)
        
        # Length filtering
        self.min_length = self.text_config.get('min_length', 10)
        self.max_length = self.text_config.get('max_length', 10000)
        
        # Language filtering
        self.allowed_languages = self.text_config.get('allowed_languages', None)
        self.detect_language = self.text_config.get('detect_language', False)
        
        # Quality filtering
        self.min_word_count = self.text_config.get('min_word_count', 3)
        self.max_repetition_ratio = self.text_config.get('max_repetition_ratio', 0.3)
        self.min_unique_words_ratio = self.text_config.get('min_unique_words_ratio', 0.3)
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info(f"Initialized TextProcessor with cleaning options: {self.text_config}")
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        # URLs
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Emails
        self.email_pattern = re.compile(
            r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        )
        
        # Phone numbers
        self.phone_pattern = re.compile(
            r'(\\+?1[-. ]?)?\\(?([0-9]{3})\\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})'
        )
        
        # Extra whitespace
        self.whitespace_pattern = re.compile(r'\\s+')
        
        # Special characters (optional)
        self.special_chars_pattern = re.compile(r'[^\\w\\s.,!?;:()\\[\\]{}"\'-]')
        
        # Quotes normalization
        self.quote_patterns = [
            (re.compile(r'["""]'), '"'),
            (re.compile(r'[''']'), "'"),
            (re.compile(r'[«»]'), '"'),
        ]
        
        # Dashes normalization
        self.dash_patterns = [
            (re.compile(r'[–—]'), '-'),
            (re.compile(r'\\s*-\\s*'), ' - '),
        ]
    
    def process_single(self, data: Any) -> Any:
        """
        Process a single text sample.
        
        Args:
            data: Input data (string or dict with text field)
            
        Returns:
            Processed data in the same format
        """
        if isinstance(data, str):
            # Direct text processing
            processed_text = self._process_text(data)
            return processed_text
        
        elif isinstance(data, dict):
            # Process dict with text fields
            processed_data = data.copy()
            
            # Find text fields to process
            text_fields = ['text', 'input', 'content', 'message', 'prompt', 'response']
            
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    processed_data[field] = self._process_text(data[field])
            
            # Validate processed data
            if not self._is_valid_sample(processed_data):
                raise ValueError("Sample failed quality validation")
            
            return processed_data
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_text(self, text: str) -> str:
        """
        Process a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        processed = text
        
        # HTML cleaning
        if self.clean_html:
            processed = self._clean_html(processed)
        
        # URL cleaning
        if self.clean_urls:
            processed = self.url_pattern.sub('[URL]', processed)
        
        # Email cleaning
        if self.clean_emails:
            processed = self.email_pattern.sub('[EMAIL]', processed)
        
        # Phone number cleaning
        if self.clean_phone_numbers:
            processed = self.phone_pattern.sub('[PHONE]', processed)
        
        # Unicode normalization
        if self.normalize_unicode:
            processed = unicodedata.normalize('NFKC', processed)
        
        # Quote normalization
        if self.normalize_quotes:
            for pattern, replacement in self.quote_patterns:
                processed = pattern.sub(replacement, processed)
        
        # Dash normalization
        if self.normalize_dashes:
            for pattern, replacement in self.dash_patterns:
                processed = pattern.sub(replacement, processed)
        
        # Case normalization
        if self.normalize_case:
            processed = processed.lower()
        
        # Special characters cleaning
        if self.clean_special_chars:
            processed = self.special_chars_pattern.sub('', processed)
        
        # Whitespace cleaning
        if self.clean_extra_whitespace:
            processed = self.whitespace_pattern.sub(' ', processed)
            processed = processed.strip()
        
        return processed
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML tags and entities from text."""
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up common HTML artifacts
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        return text
    
    def _is_valid_sample(self, data: Union[str, Dict[str, Any]]) -> bool:
        """
        Check if a sample meets quality criteria.
        
        Args:
            data: Data sample to validate
            
        Returns:
            True if sample is valid
        """
        # Extract text for validation
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            # Look for main text field
            text_fields = ['text', 'input', 'content', 'message']
            text = ""
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    break
        else:
            return False
        
        if not text:
            return False
        
        # Length validation
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Word count validation
        words = text.split()
        if len(words) < self.min_word_count:
            return False
        
        # Repetition validation
        if self._calculate_repetition_ratio(text) > self.max_repetition_ratio:
            return False
        
        # Unique words validation
        if self._calculate_unique_words_ratio(words) < self.min_unique_words_ratio:
            return False
        
        # Language validation
        if self.allowed_languages and self.detect_language:
            if not self._is_allowed_language(text):
                return False
        
        return True
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated substrings in text."""
        if len(text) < 10:
            return 0.0
        
        # Check for repeated 3-grams
        trigrams = [text[i:i+3] for i in range(len(text) - 2)]
        unique_trigrams = set(trigrams)
        
        if len(trigrams) == 0:
            return 0.0
        
        repetition_ratio = 1 - (len(unique_trigrams) / len(trigrams))
        return repetition_ratio
    
    def _calculate_unique_words_ratio(self, words: List[str]) -> float:
        """Calculate ratio of unique words to total words."""
        if len(words) == 0:
            return 0.0
        
        unique_words = set(word.lower() for word in words)
        return len(unique_words) / len(words)
    
    def _is_allowed_language(self, text: str) -> bool:
        """Check if text is in allowed language."""
        try:
            from langdetect import detect
            detected_lang = detect(text)
            return detected_lang in self.allowed_languages
        except Exception:
            # If language detection fails, assume it's allowed
            return True
    
    def _validate_input(self, data: Any) -> None:
        """Validate input data format."""
        if data is None:
            raise ValueError("Input data cannot be None")
        
        if not isinstance(data, (str, dict)):
            raise ValueError(f"Input must be string or dict, got {type(data)}")
        
        if isinstance(data, dict):
            # Check for at least one text field
            text_fields = ['text', 'input', 'content', 'message', 'prompt', 'response']
            has_text = any(field in data and isinstance(data[field], str) for field in text_fields)
            
            if not has_text:
                raise ValueError("Dict input must contain at least one text field")
    
    def _validate_output(self, data: Any) -> None:
        """Validate output data format."""
        if data is None:
            raise ValueError("Output data cannot be None")
        
        # Additional validation for processed data
        if isinstance(data, str) and len(data.strip()) == 0:
            raise ValueError("Processed text cannot be empty")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        stats = self.get_stats()
        
        # Add text-specific stats
        stats.update({
            'cleaning_options': {
                'clean_html': self.clean_html,
                'clean_urls': self.clean_urls,
                'clean_emails': self.clean_emails,
                'normalize_unicode': self.normalize_unicode,
            },
            'filtering_criteria': {
                'min_length': self.min_length,
                'max_length': self.max_length,
                'min_word_count': self.min_word_count,
                'max_repetition_ratio': self.max_repetition_ratio,
            }
        })
        
        return stats
    
    def set_quality_thresholds(self, 
                             min_length: Optional[int] = None,
                             max_length: Optional[int] = None,
                             min_word_count: Optional[int] = None,
                             max_repetition_ratio: Optional[float] = None):
        """Update quality filtering thresholds."""
        if min_length is not None:
            self.min_length = min_length
        if max_length is not None:
            self.max_length = max_length
        if min_word_count is not None:
            self.min_word_count = min_word_count
        if max_repetition_ratio is not None:
            self.max_repetition_ratio = max_repetition_ratio