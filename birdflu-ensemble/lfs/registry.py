"""Registry for labeling functions."""

import re
import numpy as np
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants for labeling function outputs
ABSTAIN = -1
HIGH_RISK = 0
MEDIUM_RISK = 1
LOW_RISK = 2
NO_RISK = 3


@dataclass
class LabelingFunction:
    """Single labeling function."""
    name: str
    func: Callable
    description: str = ""
    resources: Optional[List[str]] = None
    
    def apply(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Apply labeling function to text."""
        try:
            return self.func(text, metadata)
        except Exception as e:
            logger.warning(f"LF {self.name} failed: {e}")
            return ABSTAIN


class LabelingFunctionRegistry:
    """Registry for managing labeling functions."""
    
    def __init__(self):
        """Initialize registry."""
        self.lfs = []
        self._register_default_lfs()
    
    def register(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a labeling function.
        
        Args:
            name: Name of the LF
            description: Description of what it does
        """
        def decorator(func: Callable) -> Callable:
            lf = LabelingFunction(name=name, func=func, description=description)
            self.lfs.append(lf)
            return func
        return decorator
    
    def add_lf(self, lf: LabelingFunction):
        """Add a labeling function to registry."""
        self.lfs.append(lf)
    
    def apply_all(self, text: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Apply all labeling functions to text.
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Array of LF votes
        """
        votes = []
        for lf in self.lfs:
            vote = lf.apply(text, metadata)
            votes.append(vote)
        
        return np.array(votes)
    
    def get_lf_names(self) -> List[str]:
        """Get names of all labeling functions."""
        return [lf.name for lf in self.lfs]
    
    def _register_default_lfs(self):
        """Register default labeling functions."""
        
        # High risk patterns
        def lf_h5n1_explicit(text: str, metadata: Optional[Dict] = None) -> int:
            """Explicit H5N1 mention."""
            if re.search(r'\b[Hh]5[Nn]1\b', text):
                return HIGH_RISK
            return ABSTAIN
        
        def lf_avian_flu_outbreak(text: str, metadata: Optional[Dict] = None) -> int:
            """Avian flu outbreak mentions."""
            pattern = r'\b(avian|bird)\s+(flu|influenza)\s+(outbreak|pandemic|epidemic)\b'
            if re.search(pattern, text, re.IGNORECASE):
                return HIGH_RISK
            return ABSTAIN
        
        def lf_zoonotic_transmission(text: str, metadata: Optional[Dict] = None) -> int:
            """Zoonotic transmission mentions."""
            pattern = r'\b(zoonotic|animal.to.human|cross.species)\s+transmission'
            if re.search(pattern, text, re.IGNORECASE):
                if re.search(r'\b(flu|influenza|H5N1)\b', text, re.IGNORECASE):
                    return HIGH_RISK
            return ABSTAIN
        
        # Medium risk patterns
        def lf_bird_symptoms(text: str, metadata: Optional[Dict] = None) -> int:
            """Bird health symptoms."""
            bird_pattern = r'\b(birds?|poultry|chickens?)\b'
            symptom_pattern = r'\b(respiratory|fever|lethargy|mortality|death)\b'
            
            if re.search(bird_pattern, text, re.IGNORECASE):
                if re.search(symptom_pattern, text, re.IGNORECASE):
                    return MEDIUM_RISK
            return ABSTAIN
        
        def lf_biosecurity(text: str, metadata: Optional[Dict] = None) -> int:
            """Biosecurity measures."""
            pattern = r'\b(biosecurity|quarantine|culling)\b.*\b(birds?|poultry|farms?)\b'
            if re.search(pattern, text, re.IGNORECASE):
                return MEDIUM_RISK
            return ABSTAIN
        
        def lf_who_cdc_mention(text: str, metadata: Optional[Dict] = None) -> int:
            """WHO or CDC mentions with flu."""
            if re.search(r'\b(WHO|CDC|World Health Organization)\b', text):
                if re.search(r'\b(flu|influenza|outbreak)\b', text, re.IGNORECASE):
                    return MEDIUM_RISK
            return ABSTAIN
        
        # Low risk patterns
        def lf_general_flu(text: str, metadata: Optional[Dict] = None) -> int:
            """General flu discussion."""
            if re.search(r'\b(flu|influenza)\b', text, re.IGNORECASE):
                # Check it's not bird flu
                if not re.search(r'\b(bird|avian|H5N1|poultry)\b', text, re.IGNORECASE):
                    return LOW_RISK
            return ABSTAIN
        
        def lf_bird_health_general(text: str, metadata: Optional[Dict] = None) -> int:
            """General bird health."""
            pattern = r'\b(bird|avian)\s+(health|disease|illness)\b'
            if re.search(pattern, text, re.IGNORECASE):
                if not re.search(r'\b(flu|influenza|H5N1)\b', text, re.IGNORECASE):
                    return LOW_RISK
            return ABSTAIN
        
        # No risk patterns
        def lf_negation(text: str, metadata: Optional[Dict] = None) -> int:
            """Strong negation of bird flu."""
            pattern = r'\b(no|not|never|neither)\s+(bird\s*flu|avian\s*flu|H5N1)'
            if re.search(pattern, text, re.IGNORECASE):
                return NO_RISK
            return ABSTAIN
        
        def lf_unrelated_content(text: str, metadata: Optional[Dict] = None) -> int:
            """Clearly unrelated content."""
            # Check for common unrelated topics
            unrelated_patterns = [
                r'\b(weather|sports|entertainment|politics|technology)\b',
                r'\b(recipe|cooking|restaurant|food)\b',
                r'\b(movie|music|art|culture)\b'
            ]
            
            has_unrelated = any(re.search(p, text, re.IGNORECASE) for p in unrelated_patterns)
            has_flu = re.search(r'\b(flu|influenza|bird|avian|H5N1)\b', text, re.IGNORECASE)
            
            if has_unrelated and not has_flu:
                return NO_RISK
            return ABSTAIN
        
        # Metadata-based LFs
        def lf_news_source(text: str, metadata: Optional[Dict] = None) -> int:
            """News source credibility."""
            if metadata and metadata.get('source') == 'news':
                if re.search(r'\b(outbreak|epidemic|H5N1)\b', text, re.IGNORECASE):
                    return MEDIUM_RISK
            return ABSTAIN
        
        def lf_research_paper(text: str, metadata: Optional[Dict] = None) -> int:
            """Research paper format."""
            if re.search(r'(abstract|introduction|methods|results|discussion)', text, re.IGNORECASE):
                if re.search(r'\b(H5N1|avian\s+influenza)\b', text, re.IGNORECASE):
                    return HIGH_RISK
            return ABSTAIN
        
        # Length-based LFs
        def lf_very_short(text: str, metadata: Optional[Dict] = None) -> int:
            """Very short text - likely insufficient info."""
            if len(text.split()) < 10:
                return ABSTAIN  # Too short to classify
            return ABSTAIN
        
        def lf_contains_numbers(text: str, metadata: Optional[Dict] = None) -> int:
            """Contains statistics or case numbers."""
            number_pattern = r'\b\d{2,}\s*(cases?|deaths?|infected|confirmed)\b'
            if re.search(number_pattern, text, re.IGNORECASE):
                if re.search(r'\b(bird|avian|H5N1)\b', text, re.IGNORECASE):
                    return HIGH_RISK
                else:
                    return MEDIUM_RISK
            return ABSTAIN
        
        # Register all LFs
        self.add_lf(LabelingFunction("lf_h5n1_explicit", lf_h5n1_explicit, "Explicit H5N1 mention"))
        self.add_lf(LabelingFunction("lf_avian_flu_outbreak", lf_avian_flu_outbreak, "Avian flu outbreak"))
        self.add_lf(LabelingFunction("lf_zoonotic_transmission", lf_zoonotic_transmission, "Zoonotic transmission"))
        self.add_lf(LabelingFunction("lf_bird_symptoms", lf_bird_symptoms, "Bird health symptoms"))
        self.add_lf(LabelingFunction("lf_biosecurity", lf_biosecurity, "Biosecurity measures"))
        self.add_lf(LabelingFunction("lf_who_cdc_mention", lf_who_cdc_mention, "WHO/CDC mentions"))
        self.add_lf(LabelingFunction("lf_general_flu", lf_general_flu, "General flu discussion"))
        self.add_lf(LabelingFunction("lf_bird_health_general", lf_bird_health_general, "General bird health"))
        self.add_lf(LabelingFunction("lf_negation", lf_negation, "Negation of bird flu"))
        self.add_lf(LabelingFunction("lf_unrelated_content", lf_unrelated_content, "Unrelated content"))
        self.add_lf(LabelingFunction("lf_news_source", lf_news_source, "News source"))
        self.add_lf(LabelingFunction("lf_research_paper", lf_research_paper, "Research paper"))
        self.add_lf(LabelingFunction("lf_very_short", lf_very_short, "Very short text"))
        self.add_lf(LabelingFunction("lf_contains_numbers", lf_contains_numbers, "Contains statistics"))


# Global registry instance
registry = LabelingFunctionRegistry()