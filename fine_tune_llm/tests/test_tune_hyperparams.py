"""Comprehensive tests for hyperparameter tuning script functionality."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Test imports - Handle both script and module imports
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    
    from tune_hyperparams import (
        create_search_space, generate_hyperparameter_configs,
        evaluate_hyperparameter_config, run_hyperparameter_sweep,
        analyze_results, save_best_config
    )
    TUNE_HYPERPARAMS_AVAILABLE = True
except ImportError:
    TUNE_HYPERPARAMS_AVAILABLE = False
    pytest.skip("Tune hyperparams script not available", allow_module_level=True)


@pytest.fixture
def temp_experiment_dir():
    """Fixture providing temporary experiment directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def base_config():
    """Fixture providing base configuration for hyperparameter tuning."""
    return {
        "model": {
            "base_model_id": "ZHIPU-AI/glm-4-9b-chat",
            "model_type": "glm"
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["query_key_value", "dense"]
        },
        "training": {
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "data": {
            "max_length": 2048,
            "train_split": 0.8,
            "validation_split": 0.2
        },
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "cosine",
            "fp16": True,
            "gradient_checkpointing": True
        }
    }


@pytest.fixture
def search_space():
    """Fixture providing hyperparameter search space."""
    return {
        "lora.r": {"type": "choice", "values": [8, 16, 32, 64]},
        "lora.alpha": {"type": "choice", "values": [16, 32, 64, 128]},
        "lora.dropout": {"type": "uniform", "min": 0.05, "max": 0.2},
        "training.learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-4},
        "training.batch_size": {"type": "choice", "values": [2, 4, 8]},
        "training.warmup_ratio": {"type": "uniform", "min": 0.01, "max": 0.1},
        "training.weight_decay": {"type": "loguniform", "min": 1e-3, "max": 1e-1}
    }


@pytest.fixture
def mock_training_results():
    """Fixture providing mock training results."""
    return {
        "train_loss": 0.5,
        "eval_loss": 0.6,
        "eval_accuracy": 0.85,
        "eval_f1": 0.82,
        "eval_precision": 0.88,
        "eval_recall": 0.77,
        "training_time": 3600,  # 1 hour
        "memory_usage": 16.5,   # GB
        "convergence_epoch": 2
    }


class TestCreateSearchSpace:
    """Test search space creation functionality."""
    
    def test_create_search_space_basic(self):
        """Test basic search space creation."""
        space = create_search_space("basic")
        
        assert isinstance(space, dict)
        assert len(space) > 0
        
        # Check that each parameter has proper structure
        for param, config in space.items():
            assert "type" in config
            assert config["type"] in ["choice", "uniform", "loguniform", "normal"]
    
    def test_create_search_space_comprehensive(self):
        """Test comprehensive search space."""
        space = create_search_space("comprehensive")
        
        assert isinstance(space, dict)
        
        # Should include LoRA parameters
        lora_params = [k for k in space.keys() if k.startswith("lora.")]
        assert len(lora_params) > 0
        
        # Should include training parameters
        training_params = [k for k in space.keys() if k.startswith("training.")]
        assert len(training_params) > 0
    
    def test_create_search_space_custom(self):
        """Test custom search space creation."""
        custom_params = {
            "lora.r": {"type": "choice", "values": [4, 8, 16]},
            "training.learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-3}
        }
        
        space = create_search_space("custom", custom_params=custom_params)
        
        assert space == custom_params
    
    def test_create_search_space_invalid_type(self):
        """Test search space creation with invalid type."""
        with pytest.raises(ValueError):
            create_search_space("invalid_type")


class TestGenerateHyperparameterConfigs:
    """Test hyperparameter configuration generation."""
    
    def test_generate_configs_random(self, search_space, base_config):
        """Test random hyperparameter configuration generation."""
        configs = generate_hyperparameter_configs(
            base_config, 
            search_space, 
            method="random", 
            num_configs=10
        )
        
        assert len(configs) == 10
        assert all(isinstance(config, dict) for config in configs)
        
        # Check that each config is different from base
        for config in configs:
            assert config != base_config
    
    def test_generate_configs_grid(self, base_config):
        """Test grid search configuration generation."""
        small_space = {
            "lora.r": {"type": "choice", "values": [8, 16]},
            "lora.alpha": {"type": "choice", "values": [16, 32]}
        }
        
        configs = generate_hyperparameter_configs(
            base_config,
            small_space,
            method="grid",
            num_configs=4
        )
        
        assert len(configs) == 4  # 2 x 2 grid
        
        # Check that all combinations are present
        r_values = [config["lora"]["r"] for config in configs]
        alpha_values = [config["lora"]["alpha"] for config in configs]
        
        assert 8 in r_values and 16 in r_values
        assert 16 in alpha_values and 32 in alpha_values
    
    def test_generate_configs_bayesian(self, search_space, base_config):
        """Test Bayesian optimization configuration generation."""
        # Mock previous results for Bayesian optimization
        previous_results = [
            {"config": {"lora.r": 16, "training.learning_rate": 2e-4}, "score": 0.85},
            {"config": {"lora.r": 32, "training.learning_rate": 1e-4}, "score": 0.82}
        ]
        
        configs = generate_hyperparameter_configs(
            base_config,
            search_space,
            method="bayesian",
            num_configs=5,
            previous_results=previous_results
        )
        
        assert len(configs) <= 5  # May be less due to optimization
        assert all(isinstance(config, dict) for config in configs)
    
    def test_generate_configs_adaptive(self, search_space, base_config):
        """Test adaptive configuration generation."""
        configs = generate_hyperparameter_configs(
            base_config,
            search_space,
            method="adaptive",
            num_configs=8
        )
        
        assert len(configs) <= 8
        assert all(isinstance(config, dict) for config in configs)
    
    def test_generate_configs_invalid_method(self, search_space, base_config):
        """Test configuration generation with invalid method."""
        with pytest.raises(ValueError):
            generate_hyperparameter_configs(
                base_config,
                search_space,
                method="invalid_method"
            )


class TestEvaluateHyperparameterConfig:
    """Test hyperparameter configuration evaluation."""
    
    @patch('tune_hyperparams.train_model_with_config')
    def test_evaluate_config_success(self, mock_train, mock_training_results, base_config):
        """Test successful configuration evaluation."""
        mock_train.return_value = mock_training_results
        
        result = evaluate_hyperparameter_config(
            base_config,
            train_data=[],
            val_data=[],
            device="cpu"
        )
        
        assert isinstance(result, dict)
        assert "config" in result
        assert "metrics" in result
        assert "score" in result
        assert "training_time" in result
    
    @patch('tune_hyperparams.train_model_with_config')
    def test_evaluate_config_failure(self, mock_train, base_config):
        """Test configuration evaluation failure."""
        mock_train.side_effect = Exception("Training failed")
        
        result = evaluate_hyperparameter_config(
            base_config,
            train_data=[],
            val_data=[],
            device="cpu"
        )
        
        assert result["score"] == 0.0
        assert "error" in result
    
    @patch('tune_hyperparams.train_model_with_config')
    def test_evaluate_config_different_metrics(self, mock_train, base_config):
        """Test evaluation with different metrics."""
        # Test different optimization objectives
        objectives = ["eval_loss", "eval_accuracy", "eval_f1"]
        
        for objective in objectives:
            mock_train.return_value = {
                "eval_loss": 0.4,
                "eval_accuracy": 0.9,
                "eval_f1": 0.88
            }
            
            result = evaluate_hyperparameter_config(
                base_config,
                train_data=[],
                val_data=[],
                device="cpu",
                objective=objective
            )
            
            assert result["score"] > 0
            assert result["objective"] == objective
    
    @patch('tune_hyperparams.train_model_with_config')
    def test_evaluate_config_with_constraints(self, mock_train, base_config):
        """Test evaluation with resource constraints."""
        # Mock results that exceed memory limit
        mock_train.return_value = {
            "eval_accuracy": 0.95,
            "memory_usage": 32.0,  # Exceeds limit
            "training_time": 7200   # Exceeds limit
        }
        
        constraints = {
            "max_memory_gb": 16,
            "max_training_hours": 2
        }
        
        result = evaluate_hyperparameter_config(
            base_config,
            train_data=[],
            val_data=[],
            device="cpu",
            constraints=constraints
        )
        
        # Should penalize for exceeding constraints
        assert result["score"] < 0.95  # Penalty applied
        assert "constraint_violations" in result


class TestRunHyperparameterSweep:
    """Test hyperparameter sweep execution."""
    
    @patch('tune_hyperparams.evaluate_hyperparameter_config')
    def test_run_sweep_sequential(self, mock_evaluate, search_space, base_config, temp_experiment_dir):
        """Test sequential hyperparameter sweep."""
        # Mock evaluation results
        mock_evaluate.side_effect = [
            {"config": base_config, "score": 0.85, "metrics": {"eval_accuracy": 0.85}},
            {"config": base_config, "score": 0.88, "metrics": {"eval_accuracy": 0.88}},
            {"config": base_config, "score": 0.82, "metrics": {"eval_accuracy": 0.82}}
        ]
        
        results = run_hyperparameter_sweep(
            base_config,
            search_space,
            train_data=[],
            val_data=[],
            num_trials=3,
            method="random",
            parallel=False,
            save_dir=temp_experiment_dir
        )
        
        assert len(results) == 3
        assert all("score" in result for result in results)
        
        # Check that results are sorted by score
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @patch('tune_hyperparams.evaluate_hyperparameter_config')
    @patch('tune_hyperparams.multiprocessing.Pool')
    def test_run_sweep_parallel(self, mock_pool, mock_evaluate, search_space, base_config, temp_experiment_dir):
        """Test parallel hyperparameter sweep."""
        # Mock parallel execution
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [
            {"score": 0.85}, {"score": 0.88}, {"score": 0.82}
        ]
        
        results = run_hyperparameter_sweep(
            base_config,
            search_space,
            train_data=[],
            val_data=[],
            num_trials=3,
            method="random",
            parallel=True,
            num_workers=2,
            save_dir=temp_experiment_dir
        )
        
        assert len(results) == 3
        mock_pool.assert_called_with(2)
    
    @patch('tune_hyperparams.evaluate_hyperparameter_config')
    def test_run_sweep_with_early_stopping(self, mock_evaluate, search_space, base_config, temp_experiment_dir):
        """Test sweep with early stopping."""
        # Mock decreasing performance
        mock_evaluate.side_effect = [
            {"score": 0.85}, {"score": 0.83}, {"score": 0.80}, 
            {"score": 0.78}, {"score": 0.75}  # Consistently decreasing
        ]
        
        results = run_hyperparameter_sweep(
            base_config,
            search_space,
            train_data=[],
            val_data=[],
            num_trials=10,
            method="random",
            early_stopping_patience=3,
            save_dir=temp_experiment_dir
        )
        
        # Should stop early due to no improvement
        assert len(results) <= 10
    
    def test_run_sweep_save_results(self, temp_experiment_dir):
        """Test that sweep results are saved properly."""
        # Create mock results
        results = [
            {"config": {"lora.r": 16}, "score": 0.85, "trial_id": 0},
            {"config": {"lora.r": 32}, "score": 0.88, "trial_id": 1}
        ]
        
        with patch('tune_hyperparams.evaluate_hyperparameter_config') as mock_eval:
            mock_eval.side_effect = results
            
            run_hyperparameter_sweep(
                {},  # base config
                {"lora.r": {"type": "choice", "values": [16, 32]}},
                train_data=[],
                val_data=[],
                num_trials=2,
                save_dir=temp_experiment_dir
            )
            
            # Check that results are saved
            results_file = Path(temp_experiment_dir) / "hyperparameter_results.json"
            assert results_file.exists()


class TestAnalyzeResults:
    """Test results analysis functionality."""
    
    def test_analyze_results_basic(self, temp_experiment_dir):
        """Test basic results analysis."""
        # Create mock results
        results = [
            {
                "config": {"lora.r": 16, "training.learning_rate": 2e-4},
                "score": 0.85,
                "metrics": {"eval_accuracy": 0.85, "eval_f1": 0.82}
            },
            {
                "config": {"lora.r": 32, "training.learning_rate": 1e-4},
                "score": 0.88,
                "metrics": {"eval_accuracy": 0.88, "eval_f1": 0.85}
            },
            {
                "config": {"lora.r": 8, "training.learning_rate": 5e-4},
                "score": 0.80,
                "metrics": {"eval_accuracy": 0.80, "eval_f1": 0.78}
            }
        ]
        
        analysis = analyze_results(results)
        
        assert isinstance(analysis, dict)
        assert "best_config" in analysis
        assert "parameter_importance" in analysis
        assert "performance_summary" in analysis
        
        # Best config should be the highest scoring one
        assert analysis["best_config"]["score"] == 0.88
    
    def test_analyze_results_parameter_importance(self, temp_experiment_dir):
        """Test parameter importance analysis."""
        # Create results with clear parameter effects
        results = [
            {"config": {"lora.r": 16}, "score": 0.85},
            {"config": {"lora.r": 32}, "score": 0.90},  # Higher r = better
            {"config": {"lora.r": 8}, "score": 0.75},
            {"config": {"lora.r": 64}, "score": 0.88}
        ]
        
        analysis = analyze_results(results)
        
        assert "parameter_importance" in analysis
        # Should identify lora.r as important parameter
        if analysis["parameter_importance"]:
            assert any("lora.r" in str(param) for param in analysis["parameter_importance"])
    
    def test_analyze_results_convergence(self, temp_experiment_dir):
        """Test convergence analysis."""
        # Create results with convergence information
        results = [
            {"trial_id": 0, "score": 0.80, "convergence_epoch": 5},
            {"trial_id": 1, "score": 0.85, "convergence_epoch": 3},
            {"trial_id": 2, "score": 0.82, "convergence_epoch": 4}
        ]
        
        analysis = analyze_results(results)
        
        assert "convergence_analysis" in analysis or "performance_summary" in analysis
    
    def test_analyze_results_empty(self):
        """Test analysis with empty results."""
        analysis = analyze_results([])
        
        assert isinstance(analysis, dict)
        assert "error" in analysis or len(analysis) == 0
    
    @patch('tune_hyperparams.plt.savefig')
    @patch('tune_hyperparams.plt.figure')
    def test_analyze_results_with_plots(self, mock_figure, mock_savefig, temp_experiment_dir):
        """Test analysis with plot generation."""
        results = [
            {"score": 0.85, "trial_id": 0},
            {"score": 0.88, "trial_id": 1},
            {"score": 0.82, "trial_id": 2}
        ]
        
        analysis = analyze_results(results, save_plots=True, plot_dir=temp_experiment_dir)
        
        # Should attempt to create plots
        assert mock_figure.called or "plots_saved" in analysis


class TestSaveBestConfig:
    """Test best configuration saving."""
    
    def test_save_best_config_success(self, temp_experiment_dir):
        """Test successful best configuration saving."""
        best_result = {
            "config": {
                "lora": {"r": 32, "alpha": 64},
                "training": {"learning_rate": 1e-4, "batch_size": 4}
            },
            "score": 0.88,
            "metrics": {"eval_accuracy": 0.88, "eval_f1": 0.85}
        }
        
        save_path = save_best_config(best_result, temp_experiment_dir)
        
        assert Path(save_path).exists()
        
        # Verify saved content
        with open(save_path, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config["lora"]["r"] == 32
        assert saved_config["training"]["learning_rate"] == 1e-4
        assert "tuning_metadata" in saved_config
    
    def test_save_best_config_with_metadata(self, temp_experiment_dir):
        """Test saving configuration with metadata."""
        best_result = {
            "config": {"lora": {"r": 16}},
            "score": 0.85,
            "trial_id": 5,
            "training_time": 3600,
            "memory_usage": 12.5
        }
        
        save_path = save_best_config(
            best_result, 
            temp_experiment_dir,
            include_metadata=True
        )
        
        with open(save_path, 'r') as f:
            saved_config = json.load(f)
        
        assert "tuning_metadata" in saved_config
        assert saved_config["tuning_metadata"]["score"] == 0.85
        assert saved_config["tuning_metadata"]["trial_id"] == 5
    
    def test_save_best_config_create_directory(self, temp_experiment_dir):
        """Test saving with directory creation."""
        nested_dir = Path(temp_experiment_dir) / "nested" / "configs"
        
        best_result = {
            "config": {"lora": {"r": 8}},
            "score": 0.75
        }
        
        save_path = save_best_config(best_result, str(nested_dir))
        
        assert Path(save_path).exists()
        assert Path(save_path).parent.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_evaluate_config_out_of_memory(self, base_config):
        """Test handling of out-of-memory errors during evaluation."""
        with patch('tune_hyperparams.train_model_with_config') as mock_train:
            mock_train.side_effect = RuntimeError("CUDA out of memory")
            
            result = evaluate_hyperparameter_config(
                base_config,
                train_data=[],
                val_data=[],
                device="cuda"
            )
            
            assert result["score"] == 0.0
            assert "error" in result
            assert "memory" in result["error"].lower()
    
    def test_sweep_with_invalid_search_space(self, base_config):
        """Test sweep with invalid search space."""
        invalid_space = {
            "invalid.param": {"type": "invalid_type", "values": [1, 2, 3]}
        }
        
        with pytest.raises(ValueError):
            run_hyperparameter_sweep(
                base_config,
                invalid_space,
                train_data=[],
                val_data=[],
                num_trials=1
            )
    
    def test_analyze_results_with_nan_scores(self):
        """Test analysis with NaN scores."""
        results = [
            {"config": {"lora.r": 16}, "score": 0.85},
            {"config": {"lora.r": 32}, "score": float('nan')},
            {"config": {"lora.r": 8}, "score": 0.80}
        ]
        
        analysis = analyze_results(results)
        
        # Should handle NaN values gracefully
        assert "best_config" in analysis
        assert analysis["best_config"]["score"] == 0.85  # Should pick non-NaN value
    
    def test_save_config_permission_error(self, temp_experiment_dir):
        """Test saving configuration with permission errors."""
        best_result = {"config": {"lora": {"r": 16}}, "score": 0.85}
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                save_best_config(best_result, temp_experiment_dir)


class TestIntegration:
    """Test integration of hyperparameter tuning workflow."""
    
    @patch('tune_hyperparams.train_model_with_config')
    def test_full_tuning_pipeline(self, mock_train, search_space, base_config, temp_experiment_dir):
        """Test complete hyperparameter tuning pipeline."""
        # Mock training results with different scores
        mock_train.side_effect = [
            {"eval_accuracy": 0.85, "eval_f1": 0.82, "training_time": 3600},
            {"eval_accuracy": 0.88, "eval_f1": 0.85, "training_time": 3800},
            {"eval_accuracy": 0.82, "eval_f1": 0.79, "training_time": 3200}
        ]
        
        # Run hyperparameter sweep
        results = run_hyperparameter_sweep(
            base_config,
            search_space,
            train_data=[{"text": "sample", "label": "relevant"}],
            val_data=[{"text": "test", "label": "relevant"}],
            num_trials=3,
            method="random",
            save_dir=temp_experiment_dir
        )
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Save best configuration
        save_path = save_best_config(analysis["best_config"], temp_experiment_dir)
        
        # Verify complete pipeline
        assert len(results) == 3
        assert "best_config" in analysis
        assert Path(save_path).exists()
        
        # Check that best config has highest score
        best_score = analysis["best_config"]["score"]
        all_scores = [r["score"] for r in results]
        assert best_score == max(all_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])