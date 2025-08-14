"""
Procedural alignment verification for high-stakes predictions.

This module ensures model predictions align with established procedures,
regulations, and domain-specific requirements.
"""

import yaml
import re
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging

from ...core.interfaces import BaseComponent
from ...core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class ProceduralAlignment(BaseComponent):
    """Ensure predictions align with established procedures."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize procedural alignment component."""
        self.config = config or {}
        self.procedural_config = self.config.get('high_stakes', {}).get('procedural', {})
        self.procedures_path = self.procedural_config.get('procedures_path', 'configs/procedures.yaml')
        self.require_compliance = self.procedural_config.get('require_compliance', True)
        self.compliance_threshold = self.procedural_config.get('compliance_threshold', 0.8)
        
        self.procedures: Dict[str, Any] = {}
        self.compliance_log = []
        
        # Load procedures if available
        self._load_procedures()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config.update(config)
        self.procedural_config = self.config.get('high_stakes', {}).get('procedural', {})
        self._load_procedures()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.compliance_log.clear()
        self.procedures.clear()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "ProceduralAlignment"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def _load_procedures(self) -> None:
        """Load procedures from configuration file."""
        try:
            procedures_file = Path(self.procedures_path)
            if procedures_file.exists():
                with open(procedures_file, 'r') as f:
                    self.procedures = yaml.safe_load(f) or {}
                logger.info(f"Loaded {len(self.procedures)} procedures from {procedures_file}")
            else:
                logger.warning(f"Procedures file not found: {procedures_file}")
                # Use default procedures
                self.procedures = self._get_default_procedures()
        except Exception as e:
            logger.error(f"Error loading procedures: {e}")
            self.procedures = self._get_default_procedures()
    
    def _get_default_procedures(self) -> Dict[str, Any]:
        """Get default procedures for common domains."""
        return {
            'medical': {
                'required_elements': ['diagnosis', 'treatment', 'contraindications'],
                'prohibited_terms': ['guarantee', 'cure', 'miracle'],
                'compliance_rules': [
                    'Must include disclaimer for medical advice',
                    'Must reference evidence-based sources',
                    'Must consider patient safety'
                ]
            },
            'legal': {
                'required_elements': ['jurisdiction', 'applicable_law', 'disclaimer'],
                'prohibited_terms': ['guaranteed outcome', 'sure win'],
                'compliance_rules': [
                    'Must include legal disclaimer',
                    'Must reference relevant statutes',
                    'Must acknowledge jurisdictional limits'
                ]
            },
            'financial': {
                'required_elements': ['risk_disclosure', 'disclaimer', 'regulatory_compliance'],
                'prohibited_terms': ['guaranteed returns', 'no risk', 'sure profit'],
                'compliance_rules': [
                    'Must include investment risk disclaimer',
                    'Must comply with SEC regulations',
                    'Must avoid misleading claims'
                ]
            },
            'general': {
                'required_elements': [],
                'prohibited_terms': [],
                'compliance_rules': [
                    'Must be factually accurate',
                    'Must avoid harmful content',
                    'Must respect privacy'
                ]
            }
        }
    
    def check_compliance(self, 
                        text: str,
                        domain: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if text complies with procedural requirements.
        
        Args:
            text: Text to check
            domain: Domain for specific procedures
            metadata: Additional context
            
        Returns:
            Compliance check results
        """
        results = {
            'is_compliant': True,
            'compliance_score': 1.0,
            'violations': [],
            'warnings': [],
            'domain': domain or 'general'
        }
        
        try:
            # Determine applicable procedures
            if domain and domain in self.procedures:
                procedures = self.procedures[domain]
            else:
                procedures = self.procedures.get('general', {})
            
            # Check required elements
            required = procedures.get('required_elements', [])
            missing_elements = self._check_required_elements(text, required)
            if missing_elements:
                results['violations'].extend([
                    f"Missing required element: {element}" for element in missing_elements
                ])
                results['is_compliant'] = False
            
            # Check prohibited terms
            prohibited = procedures.get('prohibited_terms', [])
            found_prohibited = self._check_prohibited_terms(text, prohibited)
            if found_prohibited:
                results['violations'].extend([
                    f"Contains prohibited term: {term}" for term in found_prohibited
                ])
                results['is_compliant'] = False
            
            # Check compliance rules
            rules = procedures.get('compliance_rules', [])
            rule_violations = self._check_compliance_rules(text, rules, metadata)
            if rule_violations:
                results['warnings'].extend(rule_violations)
            
            # Calculate compliance score
            total_checks = len(required) + len(prohibited) + len(rules)
            if total_checks > 0:
                violations_count = len(missing_elements) + len(found_prohibited) + len(rule_violations)
                results['compliance_score'] = max(0, 1 - (violations_count / total_checks))
            
            # Determine final compliance
            if results['compliance_score'] < self.compliance_threshold:
                results['is_compliant'] = False
            
            # Log compliance check
            self.compliance_log.append({
                'text_snippet': text[:200],
                'domain': results['domain'],
                'is_compliant': results['is_compliant'],
                'score': results['compliance_score'],
                'violations': results['violations']
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            results['error'] = str(e)
            results['is_compliant'] = False
            return results
    
    def _check_required_elements(self, text: str, required_elements: List[str]) -> List[str]:
        """Check for required elements in text."""
        missing = []
        text_lower = text.lower()
        
        for element in required_elements:
            # Simple keyword matching (can be enhanced)
            element_keywords = element.lower().replace('_', ' ').split()
            if not any(keyword in text_lower for keyword in element_keywords):
                missing.append(element)
        
        return missing
    
    def _check_prohibited_terms(self, text: str, prohibited_terms: List[str]) -> List[str]:
        """Check for prohibited terms in text."""
        found = []
        text_lower = text.lower()
        
        for term in prohibited_terms:
            if term.lower() in text_lower:
                found.append(term)
        
        return found
    
    def _check_compliance_rules(self, 
                               text: str,
                               rules: List[str],
                               metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Check compliance with specific rules."""
        violations = []
        
        for rule in rules:
            rule_lower = rule.lower()
            
            # Check for disclaimer requirements
            if 'disclaimer' in rule_lower:
                if 'disclaimer' not in text.lower() and 'disclosure' not in text.lower():
                    violations.append(f"Rule violation: {rule}")
            
            # Check for evidence/source requirements
            elif 'evidence' in rule_lower or 'source' in rule_lower or 'reference' in rule_lower:
                # Simple check for citations or references
                citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'according to']
                has_citation = any(re.search(pattern, text) for pattern in citation_patterns)
                if not has_citation:
                    violations.append(f"Rule warning: {rule}")
            
            # Check for safety considerations
            elif 'safety' in rule_lower:
                safety_keywords = ['safe', 'risk', 'caution', 'warning', 'adverse']
                has_safety = any(keyword in text.lower() for keyword in safety_keywords)
                if not has_safety:
                    violations.append(f"Rule warning: {rule}")
        
        return violations
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for analysis."""
        # Simple term extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        
        key_terms = {word for word in words if len(word) > 3 and word not in stop_words}
        
        return key_terms
    
    def enhance_with_procedures(self, 
                               text: str,
                               domain: Optional[str] = None) -> str:
        """
        Enhance text with required procedural elements.
        
        Args:
            text: Original text
            domain: Domain for procedures
            
        Returns:
            Enhanced text with procedural elements
        """
        enhanced_text = text
        
        try:
            # Check compliance first
            compliance = self.check_compliance(text, domain)
            
            if not compliance['is_compliant']:
                # Add missing required elements
                if domain and domain in self.procedures:
                    procedures = self.procedures[domain]
                    
                    # Add disclaimer if needed
                    if 'disclaimer' in str(procedures.get('required_elements', [])):
                        if 'disclaimer' not in text.lower():
                            disclaimer = self._get_domain_disclaimer(domain)
                            enhanced_text = f"{enhanced_text}\n\n{disclaimer}"
                    
                    # Add other required elements as needed
                    for violation in compliance.get('violations', []):
                        if 'Missing required element' in violation:
                            element = violation.split(': ')[-1]
                            enhancement = self._get_element_template(domain, element)
                            if enhancement:
                                enhanced_text = f"{enhanced_text}\n\n{enhancement}"
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error enhancing with procedures: {e}")
            return text
    
    def _get_domain_disclaimer(self, domain: str) -> str:
        """Get appropriate disclaimer for domain."""
        disclaimers = {
            'medical': "MEDICAL DISCLAIMER: This information is for educational purposes only and should not replace professional medical advice.",
            'legal': "LEGAL DISCLAIMER: This information is for general guidance only and does not constitute legal advice.",
            'financial': "FINANCIAL DISCLAIMER: This is not financial advice. Investment decisions should be made after consulting with qualified professionals.",
            'general': "DISCLAIMER: This information is provided as-is without warranties."
        }
        
        return disclaimers.get(domain, disclaimers['general'])
    
    def _get_element_template(self, domain: str, element: str) -> Optional[str]:
        """Get template text for missing element."""
        templates = {
            'medical': {
                'contraindications': "Contraindications: Consult healthcare provider for individual assessment.",
                'treatment': "Treatment: Should be determined by qualified medical professionals.",
            },
            'legal': {
                'jurisdiction': "Jurisdiction: Laws vary by location. Consult local regulations.",
                'applicable_law': "Applicable Law: Subject to relevant statutory and case law.",
            },
            'financial': {
                'risk_disclosure': "Risk Disclosure: All investments carry risk of loss.",
                'regulatory_compliance': "Regulatory Compliance: Subject to applicable financial regulations.",
            }
        }
        
        domain_templates = templates.get(domain, {})
        return domain_templates.get(element)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate procedural compliance report."""
        report = {
            'summary': {
                'total_checks': len(self.compliance_log),
                'compliant_count': sum(1 for log in self.compliance_log if log['is_compliant']),
                'average_score': sum(log['score'] for log in self.compliance_log) / max(len(self.compliance_log), 1),
                'domains_checked': list(set(log['domain'] for log in self.compliance_log))
            },
            'violations_summary': self._summarize_violations(),
            'detailed_log': self.compliance_log[-100:],  # Last 100 entries
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_violations(self) -> Dict[str, int]:
        """Summarize violations by type."""
        violations_count = {}
        
        for log in self.compliance_log:
            for violation in log.get('violations', []):
                violation_type = violation.split(':')[0]
                violations_count[violation_type] = violations_count.get(violation_type, 0) + 1
        
        return violations_count
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on compliance analysis."""
        recommendations = []
        
        if self.compliance_log:
            compliance_rate = sum(1 for log in self.compliance_log if log['is_compliant']) / len(self.compliance_log)
            
            if compliance_rate < 0.8:
                recommendations.append(
                    f"Low compliance rate ({compliance_rate:.1%}). Review and update procedural training."
                )
            
            # Check for common violations
            violations_summary = self._summarize_violations()
            for violation_type, count in violations_summary.items():
                if count > 5:
                    recommendations.append(
                        f"Frequent {violation_type} violations ({count} occurrences). "
                        f"Consider targeted training for this requirement."
                    )
        
        if not recommendations:
            recommendations.append("Procedural compliance is satisfactory. Continue monitoring.")
        
        return recommendations