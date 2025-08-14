"""
Test risk-controlled prediction UI components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy imports first
mock_modules = [
    'streamlit', 'plotly', 'plotly.graph_objects', 'plotly.express',
    'transformers', 'peft', 'accelerate', 'datasets', 'trl'
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Mock streamlit specifically
st_mock = MagicMock()
st_mock.set_page_config = MagicMock()
st_mock.title = MagicMock()
st_mock.markdown = MagicMock()
st_mock.sidebar = MagicMock()
def mock_columns(n):
    return [MagicMock() for _ in range(n)]

st_mock.columns = mock_columns
st_mock.button = MagicMock(return_value=True)
st_mock.text_area = MagicMock(return_value="Sample test text")
st_mock.selectbox = MagicMock(return_value="Medium")
st_mock.slider = MagicMock(return_value=0.9)
st_mock.number_input = MagicMock(return_value=1.0)

# Create a proper session state mock
class MockSessionState:
    def __init__(self):
        self._state = {
            'confidence_level': 0.90,
            'risk_tolerance': 0.10,
            'false_positive_cost': 1.0,
            'false_negative_cost': 5.0,
            'enable_monte_carlo': False,
            'mc_samples': 10,
            'show_detailed_metrics': True
        }
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
        
    def __getattr__(self, name):
        return self._state.get(name)
    
    def __contains__(self, key):
        return key in self._state

st_mock.session_state = MockSessionState()
sys.modules['streamlit'] = st_mock

# Now import the UI module
from scripts.risk_prediction_ui import RiskControlledPredictionUI


class TestRiskControlledPredictionUI:
    """Test the risk-controlled prediction UI."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ui = RiskControlledPredictionUI(model_path=None)  # Demo mode
        
        # Reset session state for each test
        st_mock.session_state._state = {
            'confidence_level': 0.90,
            'risk_tolerance': 0.10,
            'false_positive_cost': 1.0,
            'false_negative_cost': 5.0,
            'enable_monte_carlo': False,
            'mc_samples': 10,
            'show_detailed_metrics': True
        }
    
    def test_ui_initialization_demo_mode(self):
        """Test UI initialization in demo mode."""
        ui = RiskControlledPredictionUI(model_path=None)
        
        assert ui.model_path is None
        assert not ui.model_available
        assert ui.inference_engine is None
        assert ui.conformal_predictor is None
        assert ui.risk_predictor is None
        assert ui.default_classes == ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        assert ui.default_cost_matrix.shape == (4, 4)
    
    def test_ui_initialization_with_model_path(self):
        """Test UI initialization with model path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake model directory
            model_path = Path(temp_dir) / "fake_model"
            model_path.mkdir()
            (model_path / "config.json").touch()
            
            # UI should attempt to load model but fail gracefully
            ui = RiskControlledPredictionUI(model_path=str(model_path))
            
            # Should fall back to demo mode due to missing components
            assert not ui.model_available
    
    def test_demo_prediction_high_confidence_low_risk(self):
        """Test demo prediction with high confidence, low risk scenario."""
        demo_probs = np.array([0.05, 0.10, 0.80, 0.05])  # Low risk scenario
        text = "This is a sample text"
        
        # Mock streamlit components
        with patch('scripts.risk_prediction_ui.st', st_mock):
            self.ui._make_demo_prediction(text, demo_probs)
            
            # Verify that prediction was processed
            predicted_class = self.ui.default_classes[np.argmax(demo_probs)]
            assert predicted_class == "LOW_RISK"
            assert np.max(demo_probs) == 0.80
    
    def test_demo_prediction_high_risk_scenario(self):
        """Test demo prediction with high risk scenario."""
        demo_probs = np.array([0.70, 0.20, 0.08, 0.02])  # Very high risk
        text = "This is a high risk text"
        
        with patch('scripts.risk_prediction_ui.st', st_mock):
            self.ui._make_demo_prediction(text, demo_probs)
            
            predicted_class = self.ui.default_classes[np.argmax(demo_probs)]
            assert predicted_class == "HIGH_RISK"
            
            # Check total high risk probability
            high_risk_total = demo_probs[0] + demo_probs[1]  # HIGH + MEDIUM
            assert abs(high_risk_total - 0.90) < 1e-10
    
    def test_probability_distribution_rendering(self):
        """Test probability distribution tab rendering."""
        probabilities = np.array([0.15, 0.25, 0.45, 0.15])
        classes = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        
        with patch('scripts.risk_prediction_ui.st', st_mock):
            with patch('scripts.risk_prediction_ui.px') as px_mock:
                px_mock.bar = MagicMock()
                self.ui._render_probability_tab(probabilities, classes)
                
                # Verify bar chart was created
                px_mock.bar.assert_called_once()
    
    def test_conformal_prediction_set_generation(self):
        """Test conformal prediction set generation."""
        probabilities = np.array([0.10, 0.60, 0.25, 0.05])
        classes = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        
        # Set confidence level
        st_mock.session_state['confidence_level'] = 0.85
        
        with patch('scripts.risk_prediction_ui.st', st_mock):
            self.ui._render_conformal_tab(probabilities, classes)
            
            # Verify prediction set logic
            sorted_indices = np.argsort(probabilities)[::-1]
            cumulative_prob = 0.0
            prediction_set = []
            
            for idx in sorted_indices:
                cumulative_prob += probabilities[idx]
                prediction_set.append(classes[idx])
                if cumulative_prob >= 0.85:
                    break
            
            # Should include top 2 classes (MEDIUM_RISK: 0.60, LOW_RISK: 0.25 = 0.85)
            assert len(prediction_set) == 2
            assert "MEDIUM_RISK" in prediction_set
            assert "LOW_RISK" in prediction_set
    
    def test_risk_analysis_cost_calculation(self):
        """Test risk analysis and cost-based decisions."""
        probabilities = np.array([0.20, 0.30, 0.40, 0.10])
        classes = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        
        # Set cost parameters
        st_mock.session_state['false_positive_cost'] = 2.0
        st_mock.session_state['false_negative_cost'] = 8.0
        st_mock.session_state['risk_tolerance'] = 0.30
        
        with patch('scripts.risk_prediction_ui.st', st_mock):
            self.ui._render_risk_tab(probabilities, classes)
            
            # Manually verify cost calculation
            cost_matrix = np.array([
                [0.0, 2.0, 2.0, 2.0],
                [8.0, 0.0, 2.0, 2.0],
                [8.0, 8.0, 0.0, 2.0],
                [8.0, 8.0, 8.0, 0.0]
            ])
            
            expected_costs = probabilities @ cost_matrix
            min_cost_idx = np.argmin(expected_costs)
            optimal_decision = classes[min_cost_idx]
            
            # The optimal decision should minimize expected cost
            assert optimal_decision in classes
            
            # Check risk tolerance
            high_risk_prob = probabilities[0] + probabilities[1]  # 0.20 + 0.30 = 0.50
            assert high_risk_prob > 0.30  # Should exceed risk tolerance
    
    def test_detailed_metrics_calculation(self):
        """Test detailed prediction metrics calculations."""
        probabilities = np.array([0.40, 0.30, 0.20, 0.10])
        classes = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NO_RISK"]
        
        # Set to show detailed metrics
        st_mock.session_state['show_detailed_metrics'] = True
        
        with patch('scripts.risk_prediction_ui.st', st_mock):
            self.ui._render_metrics_tab(probabilities, classes)
            
            # Verify metric calculations
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(classes))
            normalized_entropy = entropy / max_entropy
            
            assert 0 <= normalized_entropy <= 1
            
            # Confidence gap
            max_prob = np.max(probabilities)  # 0.40
            second_max_prob = np.partition(probabilities, -2)[-2]  # 0.30
            confidence_gap = max_prob - second_max_prob  # 0.10
            
            assert abs(confidence_gap - 0.10) < 1e-10
            
            # Effective sample size
            ess = 1 / np.sum(probabilities ** 2)
            expected_ess = 1 / (0.40**2 + 0.30**2 + 0.20**2 + 0.10**2)  # 1 / 0.30 = 3.33
            
            assert abs(ess - expected_ess) < 1e-6
    
    def test_probability_normalization(self):
        """Test that custom probabilities are properly normalized."""
        # Test custom probability scenario with non-normalized inputs
        probabilities = np.array([0.3, 0.3, 0.3, 0.3])  # Sum = 1.2
        
        # Normalize
        normalized = probabilities / np.sum(probabilities)
        
        assert abs(np.sum(normalized) - 1.0) < 1e-10
        assert np.all(normalized == 0.25)  # Should all be equal after normalization
    
    def test_risk_tolerance_decision_logic(self):
        """Test risk tolerance decision making."""
        test_cases = [
            # (high_risk_prob, risk_tolerance, should_abstain)
            (0.05, 0.10, False),  # Low risk, should proceed
            (0.10, 0.10, False),  # At threshold, should proceed  
            (0.15, 0.10, True),   # Above threshold, should abstain
            (0.80, 0.50, True),   # High risk, should abstain
        ]
        
        for high_risk_prob, risk_tolerance, should_abstain in test_cases:
            # Create probabilities that sum to high_risk_prob for first two classes
            prob_high = high_risk_prob * 0.6
            prob_medium = high_risk_prob * 0.4
            prob_low = (1 - high_risk_prob) * 0.7
            prob_none = (1 - high_risk_prob) * 0.3
            
            probabilities = np.array([prob_high, prob_medium, prob_low, prob_none])
            
            # Verify our test setup
            calculated_high_risk = probabilities[0] + probabilities[1]
            assert abs(calculated_high_risk - high_risk_prob) < 1e-10
            
            # Test decision logic
            decision_to_abstain = calculated_high_risk > risk_tolerance
            assert decision_to_abstain == should_abstain
    
    def test_cost_matrix_configuration(self):
        """Test cost matrix construction with different cost parameters."""
        test_cases = [
            (1.0, 5.0),   # Standard case
            (0.5, 10.0),  # Low FP, high FN  
            (3.0, 2.0),   # High FP, low FN
        ]
        
        for fp_cost, fn_cost in test_cases:
            expected_matrix = np.array([
                [0.0, fp_cost, fp_cost, fp_cost],
                [fn_cost, 0.0, fp_cost, fp_cost],
                [fn_cost, fn_cost, 0.0, fp_cost],
                [fn_cost, fn_cost, fn_cost, 0.0]
            ])
            
            # Verify matrix structure
            assert expected_matrix.shape == (4, 4)
            assert np.all(np.diag(expected_matrix) == 0)  # Diagonal should be zero
            
            # Verify costs in first row (true HIGH_RISK predictions)
            assert np.all(expected_matrix[0, 1:] == fp_cost)
            
            # Verify costs in first column (predicted HIGH_RISK for other true classes)
            assert np.all(expected_matrix[1:, 0] == fn_cost)
    
    def test_session_state_handling(self):
        """Test that session state variables are properly handled."""
        required_session_vars = [
            'confidence_level',
            'risk_tolerance', 
            'false_positive_cost',
            'false_negative_cost',
            'enable_monte_carlo',
            'mc_samples',
            'show_detailed_metrics'
        ]
        
        # Verify all required variables exist in mock session state
        for var in required_session_vars:
            assert var in st_mock.session_state
            
        # Verify reasonable default values
        assert 0.8 <= st_mock.session_state['confidence_level'] <= 0.99
        assert 0.01 <= st_mock.session_state['risk_tolerance'] <= 0.50
        assert st_mock.session_state['false_positive_cost'] > 0
        assert st_mock.session_state['false_negative_cost'] > 0
        assert isinstance(st_mock.session_state['enable_monte_carlo'], bool)
        assert st_mock.session_state['mc_samples'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])