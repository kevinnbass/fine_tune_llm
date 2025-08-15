"""
Complete workflow integration tests for end-to-end platform functionality.

This test module validates complete workflows from data loading through model
training, evaluation, and deployment across the entire platform.
"""

import pytest
import asyncio
import time
import threading
import tempfile
import json
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockTransformerModel, MockTokenizer, MockTrainingDataset
)

# Import platform components
from src.fine_tune_llm.core.events import EventBus, Event, EventType
from src.fine_tune_llm.config.manager import ConfigManager
from src.fine_tune_llm.config.pipeline_config import PipelineConfigManager


class TestCompleteTrainingWorkflow:
    """Test complete training workflow from start to finish."""
    
    def test_end_to_end_training_workflow(self):
        """Test complete end-to-end training workflow."""
        with mock_dependencies_context() as env:
            # Initialize workflow components
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            hf_api = env.get_service('huggingface')
            mlflow = env.get_service('mlflow')
            s3_service = env.get_service('s3')
            prometheus = env.get_infrastructure('prometheus_collector')
            
            # Workflow tracking
            workflow_events = []
            
            def track_workflow_event(event):
                workflow_events.append({
                    "type": event.type,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "source": event.source
                })
            
            # Subscribe to all relevant events
            event_types = [
                EventType.TRAINING_STARTED,
                EventType.TRAINING_EPOCH_COMPLETED,
                EventType.TRAINING_COMPLETED,
                EventType.MODEL_SAVED,
                EventType.EVALUATION_COMPLETED,
                EventType.CONFIGURATION_CHANGED
            ]
            
            for event_type in event_types:
                event_bus.subscribe(event_type, track_workflow_event)
            
            # Phase 1: Configuration Setup
            training_config = {
                "data.dataset_name": "imdb",
                "data.train_split": "train[:8000]",
                "data.val_split": "train[8000:9000]",
                "data.test_split": "test[:1000]",
                "model.name": "distilbert-base-uncased",
                "model.num_labels": 2,
                "training.learning_rate": 2e-4,
                "training.batch_size": 32,
                "training.epochs": 3,
                "training.warmup_steps": 500,
                "training.output_dir": "/tmp/training_output",
                "evaluation.batch_size": 64,
                "evaluation.metrics": ["accuracy", "f1", "precision", "recall"],
                "monitoring.log_steps": 100,
                "monitoring.eval_steps": 500,
                "experiment.name": "sentiment_classification_workflow",
                "experiment.tags": {"task": "classification", "model": "distilbert"}
            }
            
            for path, value in training_config.items():
                config_manager.set(path, value)
            
            # Phase 2: Data Loading and Preprocessing
            event_bus.publish(Event(
                EventType.DATA_LOADING_STARTED,
                {"dataset": "imdb", "splits": ["train", "validation", "test"]},
                "data_loader"
            ))
            
            # Load dataset
            dataset = hf_api.load_dataset("imdb", split="train[:9000]")
            tokenizer = hf_api.load_tokenizer("distilbert-base-uncased")
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset = MockTrainingDataset("imdb_train", size=train_size, tokenizer=tokenizer)
            val_dataset = MockTrainingDataset("imdb_val", size=val_size, tokenizer=tokenizer)
            test_dataset = MockTrainingDataset("imdb_test", size=1000, tokenizer=tokenizer)
            
            event_bus.publish(Event(
                EventType.DATA_LOADING_COMPLETED,
                {
                    "train_size": train_size,
                    "val_size": val_size,
                    "test_size": 1000,
                    "preprocessing_time": 45.2
                },
                "data_loader"
            ))
            
            # Phase 3: Model Setup and Training
            # Initialize MLFlow experiment
            experiment_id = mlflow.create_experiment("sentiment_classification_workflow")
            run_id = mlflow.start_run(experiment_id)
            
            # Log hyperparameters
            hyperparams = {
                "model_name": config_manager.get("model.name"),
                "learning_rate": config_manager.get("training.learning_rate"),
                "batch_size": config_manager.get("training.batch_size"),
                "epochs": config_manager.get("training.epochs"),
                "warmup_steps": config_manager.get("training.warmup_steps")
            }
            
            for param, value in hyperparams.items():
                mlflow.log_param(run_id, param, value)
            
            # Start training
            event_bus.publish(Event(
                EventType.TRAINING_STARTED,
                {
                    "model_name": config_manager.get("model.name"),
                    "train_size": train_size,
                    "val_size": val_size,
                    "config": hyperparams
                },
                "trainer"
            ))
            
            # Simulate training epochs
            epochs = config_manager.get("training.epochs")
            training_metrics = []
            
            for epoch in range(epochs):
                # Simulate epoch training
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": 1.5 - (epoch * 0.3),
                    "train_accuracy": 0.6 + (epoch * 0.1),
                    "val_loss": 1.4 - (epoch * 0.25),
                    "val_accuracy": 0.65 + (epoch * 0.08),
                    "learning_rate": hyperparams["learning_rate"] * (0.9 ** epoch),
                    "epoch_time": 120.5
                }
                
                training_metrics.append(epoch_metrics)
                
                # Log to MLFlow
                mlflow.log_metric(run_id, "train_loss", epoch_metrics["train_loss"], step=epoch)
                mlflow.log_metric(run_id, "train_accuracy", epoch_metrics["train_accuracy"], step=epoch)
                mlflow.log_metric(run_id, "val_loss", epoch_metrics["val_loss"], step=epoch)
                mlflow.log_metric(run_id, "val_accuracy", epoch_metrics["val_accuracy"], step=epoch)
                
                # Update Prometheus metrics
                training_counter = prometheus.counter("training_steps_total", "Training steps")
                training_counter.inc(1)
                
                loss_gauge = prometheus.gauge("current_loss", "Current training loss")
                loss_gauge.set(epoch_metrics["train_loss"])
                
                # Publish epoch completion event
                event_bus.publish(Event(
                    EventType.TRAINING_EPOCH_COMPLETED,
                    epoch_metrics,
                    "trainer"
                ))
            
            # Training completion
            final_metrics = training_metrics[-1]
            event_bus.publish(Event(
                EventType.TRAINING_COMPLETED,
                {
                    "final_train_loss": final_metrics["train_loss"],
                    "final_train_accuracy": final_metrics["train_accuracy"],
                    "final_val_loss": final_metrics["val_loss"],
                    "final_val_accuracy": final_metrics["val_accuracy"],
                    "total_epochs": epochs,
                    "total_training_time": sum(m["epoch_time"] for m in training_metrics)
                },
                "trainer"
            ))
            
            # Phase 4: Model Saving and Artifact Management
            # Save model to S3
            s3_bucket = "ml-models"
            s3_service.create_bucket(s3_bucket)
            
            model_path = f"models/{run_id}/pytorch_model.bin"
            model_data = b"Mock trained model binary data"
            
            s3_service.put_object(
                s3_bucket,
                model_path,
                model_data,
                metadata={
                    "run_id": run_id,
                    "model_name": config_manager.get("model.name"),
                    "final_accuracy": str(final_metrics["val_accuracy"]),
                    "training_date": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Log model artifact
            mlflow.log_artifact(run_id, f"s3://{s3_bucket}/{model_path}")
            
            event_bus.publish(Event(
                EventType.MODEL_SAVED,
                {
                    "model_path": f"s3://{s3_bucket}/{model_path}",
                    "model_size_mb": len(model_data) / (1024 * 1024),
                    "save_time": 15.3
                },
                "model_manager"
            ))
            
            # Phase 5: Model Evaluation
            test_metrics = {
                "test_loss": 0.85,
                "test_accuracy": 0.89,
                "test_f1": 0.88,
                "test_precision": 0.87,
                "test_recall": 0.91,
                "inference_time_ms": 45.2,
                "test_samples": 1000
            }
            
            # Log evaluation metrics
            for metric, value in test_metrics.items():
                mlflow.log_metric(run_id, metric, value)
            
            event_bus.publish(Event(
                EventType.EVALUATION_COMPLETED,
                test_metrics,
                "evaluator"
            ))
            
            # Phase 6: Model Registration and Deployment
            model_name = "sentiment-classifier"
            model_version = mlflow.register_model(
                model_uri=f"s3://{s3_bucket}/{model_path}",
                name=model_name,
                description="DistilBERT fine-tuned for sentiment classification"
            )
            
            # Update model metadata
            mlflow.update_model_version(
                name=model_name,
                version=model_version["version"],
                description=f"Model trained on {datetime.now().strftime('%Y-%m-%d')}",
                tags={
                    "accuracy": str(test_metrics["test_accuracy"]),
                    "f1_score": str(test_metrics["test_f1"]),
                    "dataset": "imdb",
                    "model_type": "distilbert"
                }
            )
            
            # Transition to staging
            mlflow.transition_model_version_stage(
                name=model_name,
                version=model_version["version"],
                stage="Staging"
            )
            
            # Phase 7: Workflow Completion
            mlflow.end_run(run_id)
            
            # Verify workflow completion
            assert len(workflow_events) >= 6  # Should have multiple workflow events
            
            # Verify training progression
            training_started_events = [e for e in workflow_events if e["type"] == EventType.TRAINING_STARTED]
            training_completed_events = [e for e in workflow_events if e["type"] == EventType.TRAINING_COMPLETED]
            epoch_completed_events = [e for e in workflow_events if e["type"] == EventType.TRAINING_EPOCH_COMPLETED]
            
            assert len(training_started_events) == 1
            assert len(training_completed_events) == 1
            assert len(epoch_completed_events) == epochs
            
            # Verify model persistence
            s3_objects = s3_service.list_objects(s3_bucket, prefix=f"models/{run_id}/")
            assert len(s3_objects) >= 1
            
            # Verify MLFlow tracking
            run_info = mlflow.get_run(run_id)
            assert run_info["status"] == "FINISHED"
            assert len(run_info["params"]) == len(hyperparams)
            assert len(run_info["metrics"]) >= len(test_metrics) + (epochs * 4)  # train/val metrics per epoch
            
            # Verify model registration
            registered_model = mlflow.get_model_version(model_name, model_version["version"])
            assert registered_model["current_stage"] == "Staging"
            assert registered_model["tags"]["accuracy"] == str(test_metrics["test_accuracy"])
    
    def test_distributed_training_workflow(self):
        """Test distributed training workflow across multiple workers."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Configure distributed training
            distributed_config = {
                "distributed.enabled": True,
                "distributed.world_size": 4,
                "distributed.backend": "nccl",
                "distributed.master_addr": "localhost",
                "distributed.master_port": "12355",
                "training.per_device_batch_size": 8,
                "training.gradient_accumulation_steps": 2,
                "data.dataloader_num_workers": 4
            }
            
            for path, value in distributed_config.items():
                config_manager.set(path, value)
            
            # Track distributed events
            distributed_events = []
            
            def track_distributed_event(event):
                distributed_events.append(event)
            
            event_bus.subscribe(EventType.DISTRIBUTED_TRAINING_STARTED, track_distributed_event)
            event_bus.subscribe(EventType.DISTRIBUTED_TRAINING_COMPLETED, track_distributed_event)
            
            # Simulate distributed training setup
            world_size = config_manager.get("distributed.world_size")
            worker_results = []
            
            def simulate_worker(rank):
                """Simulate a distributed training worker."""
                worker_config = {
                    "rank": rank,
                    "world_size": world_size,
                    "local_batch_size": config_manager.get("training.per_device_batch_size"),
                    "gradient_accumulation_steps": config_manager.get("training.gradient_accumulation_steps")
                }
                
                # Simulate worker training
                epoch_losses = []
                for epoch in range(3):
                    # Simulate distributed synchronization
                    time.sleep(0.1)  # Communication overhead
                    
                    loss = 1.5 - (epoch * 0.2) + (rank * 0.05)  # Slight variation per worker
                    epoch_losses.append(loss)
                
                worker_results.append({
                    "rank": rank,
                    "final_loss": epoch_losses[-1],
                    "epoch_losses": epoch_losses
                })
            
            # Start distributed training
            event_bus.publish(Event(
                EventType.DISTRIBUTED_TRAINING_STARTED,
                {
                    "world_size": world_size,
                    "backend": "nccl",
                    "effective_batch_size": world_size * config_manager.get("training.per_device_batch_size")
                },
                "distributed_trainer"
            ))
            
            # Simulate workers
            threads = []
            for rank in range(world_size):
                thread = threading.Thread(target=simulate_worker, args=(rank,))
                threads.append(thread)
                thread.start()
            
            # Wait for workers to complete
            for thread in threads:
                thread.join()
            
            # Aggregate results
            avg_final_loss = sum(result["final_loss"] for result in worker_results) / len(worker_results)
            
            event_bus.publish(Event(
                EventType.DISTRIBUTED_TRAINING_COMPLETED,
                {
                    "world_size": world_size,
                    "avg_final_loss": avg_final_loss,
                    "worker_results": worker_results
                },
                "distributed_trainer"
            ))
            
            # Verify distributed training
            assert len(worker_results) == world_size
            assert len(distributed_events) == 2  # Start and completion events
            
            # Verify loss convergence across workers
            final_losses = [result["final_loss"] for result in worker_results]
            loss_variance = max(final_losses) - min(final_losses)
            assert loss_variance < 0.5  # Workers should converge similarly


class TestCompleteInferenceWorkflow:
    """Test complete inference workflow from model loading to prediction."""
    
    def test_end_to_end_inference_workflow(self):
        """Test complete end-to-end inference workflow."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            hf_api = env.get_service('huggingface')
            mlflow = env.get_service('mlflow')
            s3_service = env.get_service('s3')
            prometheus = env.get_infrastructure('prometheus_collector')
            
            # Track inference events
            inference_events = []
            
            def track_inference_event(event):
                inference_events.append(event)
            
            event_bus.subscribe(EventType.INFERENCE_STARTED, track_inference_event)
            event_bus.subscribe(EventType.INFERENCE_COMPLETED, track_inference_event)
            event_bus.subscribe(EventType.MODEL_LOADED, track_inference_event)
            
            # Phase 1: Configure inference pipeline
            inference_config = {
                "inference.model_name": "sentiment-classifier",
                "inference.model_version": "1",
                "inference.batch_size": 32,
                "inference.max_length": 512,
                "inference.device": "cuda:0",
                "inference.use_fp16": True,
                "inference.use_cache": True,
                "inference.temperature": 1.0,
                "inference.confidence_threshold": 0.8,
                "inference.return_probabilities": True
            }
            
            for path, value in inference_config.items():
                config_manager.set(path, value)
            
            # Phase 2: Load model from registry
            model_name = config_manager.get("inference.model_name")
            model_version = config_manager.get("inference.model_version")
            
            # Get model info from MLFlow
            model_info = mlflow.get_model_version(model_name, model_version)
            model_uri = model_info.get("source", "s3://ml-models/models/test_run/pytorch_model.bin")
            
            # Load model from S3
            s3_bucket = "ml-models"
            s3_key = "models/test_run/pytorch_model.bin"
            
            model_data = s3_service.get_object(s3_bucket, s3_key)
            assert model_data["success"]
            
            # Load tokenizer
            tokenizer = hf_api.load_tokenizer("distilbert-base-uncased")
            model = MockTransformerModel("distilbert-base-uncased")
            
            event_bus.publish(Event(
                EventType.MODEL_LOADED,
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "model_size_mb": len(model_data["data"]) / (1024 * 1024),
                    "load_time": 2.5
                },
                "inference_engine"
            ))
            
            # Phase 3: Prepare inference data
            test_texts = [
                "This movie was absolutely fantastic! I loved every minute of it.",
                "The film was boring and poorly executed. Complete waste of time.",
                "It was an okay movie, nothing special but not terrible either.",
                "Amazing cinematography and outstanding performances by all actors.",
                "I fell asleep halfway through. Very disappointing.",
                "A masterpiece of modern cinema. Highly recommended!",
                "The plot was confusing and the ending made no sense.",
                "Good entertainment value, though a bit predictable.",
                "Worst movie I've ever seen. Avoid at all costs.",
                "A delightful film that the whole family can enjoy."
            ]
            
            # Phase 4: Run batch inference
            batch_size = config_manager.get("inference.batch_size")
            max_length = config_manager.get("inference.max_length")
            
            event_bus.publish(Event(
                EventType.INFERENCE_STARTED,
                {
                    "batch_size": len(test_texts),
                    "max_length": max_length,
                    "model_name": model_name
                },
                "inference_engine"
            ))
            
            # Process in batches
            all_predictions = []
            all_probabilities = []
            
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                
                # Tokenize batch
                encoding = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Run inference
                outputs = model(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask
                )
                
                # Process outputs
                import torch
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Convert to labels
                label_map = {0: "negative", 1: "positive"}
                batch_predictions = [label_map[pred.item()] for pred in predictions]
                batch_probabilities = probabilities.tolist()
                
                all_predictions.extend(batch_predictions)
                all_probabilities.extend(batch_probabilities)
                
                # Update metrics
                inference_counter = prometheus.counter("inference_requests_total", "Inference requests")
                inference_counter.inc(len(batch_texts))
                
                latency_histogram = prometheus.histogram("inference_latency_seconds", "Inference latency")
                latency_histogram.observe(0.1)  # Mock latency
            
            # Phase 5: Post-process results
            confidence_threshold = config_manager.get("inference.confidence_threshold")
            
            final_results = []
            for i, (text, prediction, probabilities) in enumerate(zip(test_texts, all_predictions, all_probabilities)):
                confidence = max(probabilities)
                
                result = {
                    "text": text,
                    "prediction": prediction if confidence >= confidence_threshold else "uncertain",
                    "confidence": confidence,
                    "probabilities": {
                        "negative": probabilities[0],
                        "positive": probabilities[1]
                    }
                }
                final_results.append(result)
            
            # Phase 6: Log inference results
            inference_stats = {
                "total_samples": len(test_texts),
                "high_confidence_predictions": sum(1 for r in final_results if r["confidence"] >= confidence_threshold),
                "uncertain_predictions": sum(1 for r in final_results if r["prediction"] == "uncertain"),
                "positive_predictions": sum(1 for r in final_results if r["prediction"] == "positive"),
                "negative_predictions": sum(1 for r in final_results if r["prediction"] == "negative"),
                "average_confidence": sum(r["confidence"] for r in final_results) / len(final_results),
                "total_inference_time": 1.5
            }
            
            event_bus.publish(Event(
                EventType.INFERENCE_COMPLETED,
                inference_stats,
                "inference_engine"
            ))
            
            # Verify inference workflow
            assert len(inference_events) == 3  # Model loaded, inference started, inference completed
            assert len(final_results) == len(test_texts)
            
            # Verify predictions were made
            predictions_made = [r for r in final_results if r["prediction"] != "uncertain"]
            assert len(predictions_made) >= len(test_texts) * 0.7  # At least 70% should be confident
            
            # Verify metrics were recorded
            total_requests = prometheus.get_metric_value("inference_requests_total")
            assert total_requests == len(test_texts)
    
    def test_real_time_inference_workflow(self):
        """Test real-time inference workflow with streaming."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Configure real-time inference
            realtime_config = {
                "inference.mode": "realtime",
                "inference.max_concurrent_requests": 10,
                "inference.timeout_seconds": 30,
                "inference.cache_enabled": True,
                "inference.cache_ttl_seconds": 300,
                "monitoring.enable_metrics": True,
                "monitoring.enable_logging": True
            }
            
            for path, value in realtime_config.items():
                config_manager.set(path, value)
            
            # Track real-time events
            realtime_events = []
            request_latencies = []
            
            def track_realtime_event(event):
                realtime_events.append(event)
                if "latency" in event.data:
                    request_latencies.append(event.data["latency"])
            
            event_bus.subscribe(EventType.INFERENCE_REQUEST_RECEIVED, track_realtime_event)
            event_bus.subscribe(EventType.INFERENCE_REQUEST_COMPLETED, track_realtime_event)
            
            # Simulate real-time requests
            requests = [
                {"id": "req_1", "text": "This is a great product!", "timestamp": time.time()},
                {"id": "req_2", "text": "Poor quality, not worth it.", "timestamp": time.time() + 0.1},
                {"id": "req_3", "text": "Average performance, okay value.", "timestamp": time.time() + 0.2},
                {"id": "req_4", "text": "Excellent service and fast delivery!", "timestamp": time.time() + 0.3},
                {"id": "req_5", "text": "Disappointing experience overall.", "timestamp": time.time() + 0.4}
            ]
            
            responses = []
            
            def process_request(request):
                """Simulate real-time request processing."""
                start_time = time.time()
                
                # Publish request received event
                event_bus.publish(Event(
                    EventType.INFERENCE_REQUEST_RECEIVED,
                    {
                        "request_id": request["id"],
                        "text_length": len(request["text"]),
                        "received_at": request["timestamp"]
                    },
                    "inference_api"
                ))
                
                # Simulate processing
                time.sleep(0.05)  # Mock processing time
                
                # Generate response
                response = {
                    "request_id": request["id"],
                    "prediction": "positive" if "great" in request["text"] or "excellent" in request["text"] else "negative",
                    "confidence": 0.85,
                    "processing_time": time.time() - start_time
                }
                
                responses.append(response)
                
                # Publish completion event
                event_bus.publish(Event(
                    EventType.INFERENCE_REQUEST_COMPLETED,
                    {
                        "request_id": request["id"],
                        "prediction": response["prediction"],
                        "confidence": response["confidence"],
                        "latency": response["processing_time"]
                    },
                    "inference_api"
                ))
            
            # Process requests concurrently
            threads = []
            for request in requests:
                thread = threading.Thread(target=process_request, args=(request,))
                threads.append(thread)
                thread.start()
            
            # Wait for all requests to complete
            for thread in threads:
                thread.join()
            
            # Verify real-time processing
            assert len(responses) == len(requests)
            assert len(realtime_events) == len(requests) * 2  # Received + completed for each
            
            # Verify latency requirements
            avg_latency = sum(request_latencies) / len(request_latencies)
            assert avg_latency < 0.5  # Should be fast for real-time
            
            # Verify all requests were processed
            request_ids = {response["request_id"] for response in responses}
            expected_ids = {request["id"] for request in requests}
            assert request_ids == expected_ids


class TestCompleteMLOpsWorkflow:
    """Test complete MLOps workflow including CI/CD and monitoring."""
    
    def test_model_deployment_pipeline_workflow(self):
        """Test complete model deployment pipeline workflow."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            mlflow = env.get_service('mlflow')
            s3_service = env.get_service('s3')
            hf_api = env.get_service('huggingface')
            
            # Track deployment events
            deployment_events = []
            
            def track_deployment_event(event):
                deployment_events.append(event)
            
            event_bus.subscribe(EventType.MODEL_DEPLOYMENT_STARTED, track_deployment_event)
            event_bus.subscribe(EventType.MODEL_DEPLOYMENT_COMPLETED, track_deployment_event)
            event_bus.subscribe(EventType.MODEL_VALIDATION_COMPLETED, track_deployment_event)
            
            # Phase 1: Model Selection and Validation
            model_name = "sentiment-classifier"
            
            # Get latest model from staging
            staging_models = mlflow.get_model_versions(model_name, stage="Staging")
            if not staging_models:
                # Create a staging model for testing
                model_version = mlflow.register_model(
                    model_uri="s3://ml-models/models/test/",
                    name=model_name
                )
                mlflow.transition_model_version_stage(
                    name=model_name,
                    version=model_version["version"],
                    stage="Staging"
                )
                staging_models = [model_version]
            
            candidate_model = staging_models[0]
            
            # Phase 2: Model Validation
            validation_config = {
                "validation.accuracy_threshold": 0.85,
                "validation.f1_threshold": 0.80,
                "validation.latency_threshold_ms": 100,
                "validation.test_dataset_size": 1000
            }
            
            for path, value in validation_config.items():
                config_manager.set(path, value)
            
            # Run validation tests
            validation_results = {
                "accuracy": 0.89,
                "f1_score": 0.87,
                "precision": 0.86,
                "recall": 0.88,
                "avg_latency_ms": 45.2,
                "p95_latency_ms": 78.1,
                "test_samples": 1000,
                "validation_passed": True
            }
            
            # Check validation thresholds
            validation_passed = (
                validation_results["accuracy"] >= config_manager.get("validation.accuracy_threshold") and
                validation_results["f1_score"] >= config_manager.get("validation.f1_threshold") and
                validation_results["avg_latency_ms"] <= config_manager.get("validation.latency_threshold_ms")
            )
            
            validation_results["validation_passed"] = validation_passed
            
            event_bus.publish(Event(
                EventType.MODEL_VALIDATION_COMPLETED,
                validation_results,
                "validation_service"
            ))
            
            if not validation_passed:
                pytest.fail("Model validation failed - cannot proceed with deployment")
            
            # Phase 3: Deployment Preparation
            deployment_config = {
                "deployment.environment": "production",
                "deployment.replicas": 3,
                "deployment.cpu_request": "500m",
                "deployment.memory_request": "1Gi",
                "deployment.gpu_enabled": True,
                "deployment.auto_scaling.enabled": True,
                "deployment.auto_scaling.min_replicas": 2,
                "deployment.auto_scaling.max_replicas": 10,
                "deployment.health_check.enabled": True,
                "deployment.health_check.interval_seconds": 30
            }
            
            for path, value in deployment_config.items():
                config_manager.set(path, value)
            
            # Phase 4: Model Deployment
            event_bus.publish(Event(
                EventType.MODEL_DEPLOYMENT_STARTED,
                {
                    "model_name": model_name,
                    "model_version": candidate_model["version"],
                    "environment": "production",
                    "deployment_strategy": "blue_green"
                },
                "deployment_service"
            ))
            
            # Simulate deployment steps
            deployment_steps = [
                {"step": "download_model", "duration": 30.2, "status": "completed"},
                {"step": "build_container", "duration": 120.5, "status": "completed"},
                {"step": "deploy_to_staging", "duration": 45.1, "status": "completed"},
                {"step": "run_smoke_tests", "duration": 60.3, "status": "completed"},
                {"step": "deploy_to_production", "duration": 90.7, "status": "completed"},
                {"step": "configure_load_balancer", "duration": 15.2, "status": "completed"}
            ]
            
            total_deployment_time = sum(step["duration"] for step in deployment_steps)
            
            # Phase 5: Post-deployment Validation
            post_deployment_checks = {
                "health_check_passed": True,
                "load_test_passed": True,
                "integration_test_passed": True,
                "performance_regression_check": True,
                "security_scan_passed": True
            }
            
            # Phase 6: Update Model Registry
            if all(post_deployment_checks.values()):
                # Transition to production
                mlflow.transition_model_version_stage(
                    name=model_name,
                    version=candidate_model["version"],
                    stage="Production"
                )
                
                # Update deployment metadata
                mlflow.update_model_version(
                    name=model_name,
                    version=candidate_model["version"],
                    description="Deployed to production",
                    tags={
                        "deployment_date": datetime.now(timezone.utc).isoformat(),
                        "deployment_environment": "production",
                        "validation_accuracy": str(validation_results["accuracy"]),
                        "deployment_time_seconds": str(total_deployment_time)
                    }
                )
                
                deployment_status = "success"
            else:
                deployment_status = "failed"
                
                # Rollback if needed
                mlflow.transition_model_version_stage(
                    name=model_name,
                    version=candidate_model["version"],
                    stage="Archived"
                )
            
            event_bus.publish(Event(
                EventType.MODEL_DEPLOYMENT_COMPLETED,
                {
                    "model_name": model_name,
                    "model_version": candidate_model["version"],
                    "deployment_status": deployment_status,
                    "deployment_time": total_deployment_time,
                    "deployment_steps": deployment_steps,
                    "post_deployment_checks": post_deployment_checks
                },
                "deployment_service"
            ))
            
            # Verify deployment workflow
            assert len(deployment_events) == 3  # Started, validation completed, deployment completed
            assert deployment_status == "success"
            
            # Verify model promotion
            production_model = mlflow.get_model_version(model_name, candidate_model["version"])
            assert production_model["current_stage"] == "Production"
            assert "deployment_date" in production_model["tags"]
    
    def test_continuous_monitoring_workflow(self):
        """Test continuous monitoring workflow for deployed models."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            prometheus = env.get_infrastructure('prometheus_collector')
            grafana = env.get_infrastructure('grafana_client')
            email_service = env.get_infrastructure('email_service')
            slack_notifier = env.get_infrastructure('slack_notifier')
            
            # Configure monitoring
            monitoring_config = {
                "monitoring.enabled": True,
                "monitoring.metrics_interval_seconds": 10,
                "monitoring.alert_thresholds.error_rate": 0.05,
                "monitoring.alert_thresholds.latency_p95_ms": 200,
                "monitoring.alert_thresholds.accuracy_drop": 0.10,
                "monitoring.drift_detection.enabled": True,
                "monitoring.drift_detection.window_size": 1000,
                "monitoring.drift_detection.threshold": 0.2
            }
            
            for path, value in monitoring_config.items():
                config_manager.set(path, value)
            
            # Set up monitoring metrics
            request_counter = prometheus.counter("model_requests_total", "Total model requests")
            error_counter = prometheus.counter("model_errors_total", "Total model errors")
            latency_histogram = prometheus.histogram(
                "model_latency_seconds",
                "Model inference latency",
                buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            )
            accuracy_gauge = prometheus.gauge("model_accuracy", "Current model accuracy")
            
            # Track monitoring events
            monitoring_events = []
            alerts_triggered = []
            
            def track_monitoring_event(event):
                monitoring_events.append(event)
                if event.type == EventType.ALERT_TRIGGERED:
                    alerts_triggered.append(event)
            
            event_bus.subscribe(EventType.MONITORING_STARTED, track_monitoring_event)
            event_bus.subscribe(EventType.ALERT_TRIGGERED, track_monitoring_event)
            event_bus.subscribe(EventType.DRIFT_DETECTED, track_monitoring_event)
            
            # Start monitoring
            event_bus.publish(Event(
                EventType.MONITORING_STARTED,
                {
                    "model_name": "sentiment-classifier",
                    "monitoring_config": monitoring_config
                },
                "monitoring_service"
            ))
            
            # Simulate production traffic
            def simulate_production_traffic():
                """Simulate production model traffic with various conditions."""
                # Normal operation
                for i in range(100):
                    request_counter.inc(1)
                    latency_histogram.observe(0.05)  # Normal latency
                    accuracy_gauge.set(0.89)  # Good accuracy
                
                # Gradually increasing latency (performance degradation)
                for i in range(50):
                    request_counter.inc(1)
                    latency = 0.05 + (i * 0.01)  # Increasing latency
                    latency_histogram.observe(latency)
                    
                    # Check latency threshold
                    if latency > 0.2:  # p95 threshold exceeded
                        event_bus.publish(Event(
                            EventType.ALERT_TRIGGERED,
                            {
                                "alert_type": "high_latency",
                                "metric_name": "model_latency_p95",
                                "current_value": latency * 1000,  # Convert to ms
                                "threshold": config_manager.get("monitoring.alert_thresholds.latency_p95_ms"),
                                "severity": "warning"
                            },
                            "monitoring_service"
                        ))
                
                # Some errors
                for i in range(10):
                    request_counter.inc(1)
                    error_counter.inc(1)
                    
                    current_error_rate = 10 / (100 + 50 + i + 1)  # Calculate current error rate
                    if current_error_rate > config_manager.get("monitoring.alert_thresholds.error_rate"):
                        event_bus.publish(Event(
                            EventType.ALERT_TRIGGERED,
                            {
                                "alert_type": "high_error_rate",
                                "metric_name": "model_error_rate",
                                "current_value": current_error_rate,
                                "threshold": config_manager.get("monitoring.alert_thresholds.error_rate"),
                                "severity": "critical"
                            },
                            "monitoring_service"
                        ))
                
                # Accuracy drop (data drift)
                accuracy_drop = 0.15  # 15% accuracy drop
                new_accuracy = 0.89 - accuracy_drop
                accuracy_gauge.set(new_accuracy)
                
                if accuracy_drop > config_manager.get("monitoring.alert_thresholds.accuracy_drop"):
                    event_bus.publish(Event(
                        EventType.DRIFT_DETECTED,
                        {
                            "drift_type": "accuracy_drift",
                            "baseline_accuracy": 0.89,
                            "current_accuracy": new_accuracy,
                            "accuracy_drop": accuracy_drop,
                            "drift_magnitude": accuracy_drop / 0.89,
                            "detection_method": "statistical"
                        },
                        "drift_detector"
                    ))
            
            # Run traffic simulation
            simulate_production_traffic()
            
            # Process alerts
            for alert_event in alerts_triggered:
                alert_data = alert_event.data
                
                # Send email alert for critical issues
                if alert_data.get("severity") == "critical":
                    email_service.send_email(
                        to_address="ml-team@company.com",
                        subject=f"Critical Alert: {alert_data['alert_type']}",
                        body=f"Alert: {alert_data['metric_name']} = {alert_data['current_value']}, threshold = {alert_data['threshold']}"
                    )
                
                # Send Slack notification
                slack_notifier.send_alert(
                    severity=alert_data.get("severity", "warning"),
                    title=f"Model Monitoring Alert: {alert_data['alert_type']}",
                    message=f"Metric: {alert_data['metric_name']}\nCurrent: {alert_data['current_value']}\nThreshold: {alert_data['threshold']}",
                    channel="#ml-alerts"
                )
            
            # Verify monitoring workflow
            assert len(monitoring_events) >= 1
            assert len(alerts_triggered) >= 2  # Latency and error rate alerts
            
            # Verify alert notifications
            sent_emails = email_service.get_sent_emails(hours=1)
            sent_messages = slack_notifier.get_messages(channel="#ml-alerts", hours=1)
            
            assert len(sent_emails) >= 1  # At least one critical alert email
            assert len(sent_messages) >= len(alerts_triggered)  # Slack messages for all alerts
            
            # Verify drift detection
            drift_events = [e for e in monitoring_events if e.type == EventType.DRIFT_DETECTED]
            assert len(drift_events) >= 1
            
            drift_event = drift_events[0]
            assert drift_event.data["accuracy_drop"] > config_manager.get("monitoring.alert_thresholds.accuracy_drop")
            assert drift_event.data["drift_magnitude"] > 0.1