"""MLflow experiment tracking integration."""

import os
import json
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking for bird flu ensemble."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "birdflu-ensemble",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Location to store artifacts
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        self.client = MlflowClient()
        self.current_run = None
        
        logger.info(f"Initialized MLflow tracking for experiment: {experiment_name}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Run ID
        """
        if self.current_run:
            logger.warning("Previous run still active, ending it first")
            self.end_run()
        
        # Default tags
        default_tags = {
            "mlflow.runName": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "project": "birdflu-ensemble",
            "framework": "ensemble",
        }
        
        if tags:
            default_tags.update(tags)
        
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            tags=default_tags
        )
        
        logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
        return self.current_run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Flatten nested parameters
        flat_params = self._flatten_params(params)
        
        for key, value in flat_params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics."""
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log a model to MLflow.
        
        Args:
            model: Model object
            model_name: Name for the model
            model_type: Type of model ('sklearn', 'pytorch', 'custom')
            signature: Model signature
            input_example: Example input
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            # Custom model - use pickle
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model, f)
                mlflow.log_artifact(f.name, f"models/{model_name}")
            
            os.unlink(f.name)
    
    def log_artifacts(self, artifact_path: str, local_dir: Optional[str] = None):
        """Log artifacts."""
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if local_dir:
            mlflow.log_artifacts(local_dir, artifact_path)
        else:
            mlflow.log_artifact(artifact_path)
    
    def log_voter_results(
        self,
        voter_name: str,
        metrics: Dict[str, float],
        model: Any,
        config: Dict[str, Any]
    ):
        """
        Log results for a single voter.
        
        Args:
            voter_name: Name of the voter
            metrics: Performance metrics
            model: Trained model
            config: Training configuration
        """
        # Log voter-specific metrics with prefix
        voter_metrics = {f"{voter_name}_{k}": v for k, v in metrics.items()}
        self.log_metrics(voter_metrics)
        
        # Log voter config
        voter_params = {f"{voter_name}_{k}": v for k, v in config.items()}
        self.log_params(voter_params)
        
        # Log model
        if model:
            model_type = "sklearn" if hasattr(model, 'predict') else "custom"
            self.log_model(model, f"voter_{voter_name}", model_type)
    
    def log_cascade_results(
        self,
        cascade_metrics: Dict[str, float],
        stacker_model: Any,
        conformal_thresholds: Dict[str, float]
    ):
        """
        Log cascade/stacker results.
        
        Args:
            cascade_metrics: Cascade performance metrics
            stacker_model: Trained stacker model
            conformal_thresholds: Conformal prediction thresholds
        """
        # Log cascade metrics
        cascade_metrics_prefixed = {f"cascade_{k}": v for k, v in cascade_metrics.items()}
        self.log_metrics(cascade_metrics_prefixed)
        
        # Log conformal thresholds
        threshold_params = {f"conformal_{k}": v for k, v in conformal_thresholds.items()}
        self.log_params(threshold_params)
        
        # Log stacker model
        if stacker_model:
            self.log_model(stacker_model, "stacker_model", "sklearn")
    
    def log_evaluation_results(
        self,
        overall_metrics: Dict[str, float],
        slice_metrics: Dict[str, Dict[str, float]],
        confusion_matrix: Optional[np.ndarray] = None
    ):
        """
        Log evaluation results.
        
        Args:
            overall_metrics: Overall performance metrics
            slice_metrics: Per-slice metrics
            confusion_matrix: Confusion matrix
        """
        # Log overall metrics
        self.log_metrics(overall_metrics)
        
        # Log slice metrics
        for slice_name, metrics in slice_metrics.items():
            slice_metrics_prefixed = {f"slice_{slice_name}_{k}": v 
                                    for k, v in metrics.items()}
            self.log_metrics(slice_metrics_prefixed)
        
        # Log confusion matrix as artifact
        if confusion_matrix is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                plt.savefig(f.name, bbox_inches='tight')
                self.log_artifacts(f.name, "plots")
            
            plt.close()
            os.unlink(f.name)
    
    def log_data_drift(
        self,
        drift_metrics: List[Dict[str, Any]],
        drift_plots: Optional[List[str]] = None
    ):
        """
        Log data drift detection results.
        
        Args:
            drift_metrics: List of drift detection results
            drift_plots: Optional paths to drift plots
        """
        # Aggregate drift metrics
        total_drifts = sum(1 for d in drift_metrics if d.get('is_drift', False))
        avg_drift_score = np.mean([d.get('drift_score', 0) for d in drift_metrics])
        
        self.log_metrics({
            'data_drift_count': total_drifts,
            'avg_drift_score': avg_drift_score,
            'drift_detection_timestamp': datetime.now().timestamp()
        })
        
        # Log detailed drift results as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            json.dump(drift_metrics, f, indent=2, default=str)
            self.log_artifacts(f.name, "drift")
        
        os.unlink(f.name)
        
        # Log drift plots
        if drift_plots:
            for plot_path in drift_plots:
                self.log_artifacts(plot_path, "drift_plots")
    
    def log_prediction_stats(
        self,
        prediction_count: int,
        abstention_rate: float,
        avg_latency: float,
        cost_per_prediction: float
    ):
        """
        Log prediction statistics.
        
        Args:
            prediction_count: Number of predictions made
            abstention_rate: Rate of abstentions
            avg_latency: Average prediction latency
            cost_per_prediction: Average cost per prediction
        """
        self.log_metrics({
            'prediction_count': prediction_count,
            'abstention_rate': abstention_rate,
            'avg_latency_ms': avg_latency,
            'cost_per_prediction_cents': cost_per_prediction,
            'stats_timestamp': datetime.now().timestamp()
        })
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self.current_run:
            run_status = RunStatus.to_proto(status)
            mlflow.end_run(status=run_status)
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
    
    def get_best_run(
        self,
        metric_name: str = "f1_weighted",
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            ascending: Whether lower is better
            
        Returns:
            Best run information
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            return {
                'run_id': best_run.info.run_id,
                'metrics': best_run.data.metrics,
                'params': best_run.data.params,
                'tags': best_run.data.tags,
                'status': best_run.info.status,
                'start_time': best_run.info.start_time,
                'end_time': best_run.info.end_time
            }
        
        return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple runs on specified metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                metric: run.data.metrics.get(metric, float('nan'))
                for metric in metrics
            }
        
        return comparison
    
    def load_model(self, run_id: str, model_name: str) -> Any:
        """
        Load a model from a specific run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        
        try:
            # Try sklearn first
            return mlflow.sklearn.load_model(model_uri)
        except:
            try:
                # Try pytorch
                return mlflow.pytorch.load_model(model_uri)
            except:
                # Try loading as artifact
                artifact_path = self.client.download_artifacts(run_id, f"models/{model_name}")
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        model_path: str,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            run_id: Run ID containing the model
            model_path: Path to the model within the run
            description: Optional description
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            description=description
        )
        
        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to a different stage.
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Flatten nested parameters for MLflow."""
        flat = {}
        
        for key, value in params.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_params(value, f"{new_key}_"))
            elif isinstance(value, (list, tuple)):
                flat[new_key] = str(value)
            else:
                flat[new_key] = str(value)
        
        return flat


# Context manager for experiment tracking
class ExperimentRun:
    """Context manager for MLflow experiment runs."""
    
    def __init__(
        self,
        tracker: MLflowTracker,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize experiment run context.
        
        Args:
            tracker: MLflow tracker instance
            run_name: Name for the run
            tags: Tags to add to the run
        """
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run_id = None
    
    def __enter__(self):
        """Start the run."""
        self.run_id = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the run."""
        if exc_type is not None:
            self.tracker.end_run("FAILED")
        else:
            self.tracker.end_run("FINISHED")


# Global tracker instance
mlflow_tracker = MLflowTracker()