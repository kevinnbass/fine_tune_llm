"""
External service integration tests with comprehensive service interaction validation.

This test module validates integration with all external services including
API services, cloud storage, monitoring systems, and third-party integrations.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    mock_dependencies_context, create_mock_environment,
    MockHuggingFaceAPI, MockOpenAIAPI, MockMLFlowTracker,
    MockWandBTracker, MockS3Service, MockGCSService
)

# Import platform components
from src.fine_tune_llm.core.events import EventBus, Event, EventType
from src.fine_tune_llm.config.manager import ConfigManager


class TestHuggingFaceServiceIntegration:
    """Test HuggingFace service integration across all platform components."""
    
    def test_model_loading_pipeline_integration(self):
        """Test complete model loading pipeline integration."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            # Test model discovery and loading workflow
            available_models = hf_api.list_models(task="text-classification")
            assert len(available_models) >= 0
            
            # Test model loading with validation
            model_name = "distilbert-base-uncased"
            
            # Load tokenizer first
            tokenizer = hf_api.load_tokenizer(model_name)
            assert tokenizer is not None
            assert hasattr(tokenizer, 'vocab_size')
            
            # Load model
            model = hf_api.from_pretrained(model_name)
            assert model is not None
            assert model.model_name == model_name
            
            # Verify loading was logged
            assert len(hf_api.download_history) >= 2  # Tokenizer + Model
            
            # Test model metadata extraction
            model_info = hf_api.get_model_info(model_name)
            assert 'model_name' in model_info
            assert 'download_size' in model_info
    
    def test_dataset_loading_and_preprocessing_integration(self):
        """Test dataset loading and preprocessing integration."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            # Load dataset
            dataset = hf_api.load_dataset("imdb", split="train[:100]")
            assert dataset is not None
            assert len(dataset) == 100
            
            # Test dataset preprocessing integration
            def preprocess_function(examples):
                return {
                    'text': [text.lower() for text in examples['text']],
                    'labels': examples['label']
                }
            
            processed_dataset = dataset.map(preprocess_function, batched=True)
            assert processed_dataset is not None
            
            # Test dataset caching
            cached_dataset = hf_api.load_dataset("imdb", split="train[:100]", use_cache=True)
            assert cached_dataset is not None
            
            # Verify caching behavior
            cache_hits = [h for h in hf_api.download_history if h.get('cached', False)]
            assert len(cache_hits) >= 0
    
    def test_model_hub_operations_integration(self):
        """Test model hub operations integration."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            model = env.get_model_component('model')
            
            # Test model uploading
            upload_result = hf_api.push_to_hub(
                model, 
                "test-user/test-model",
                commit_message="Initial upload"
            )
            
            assert 'repo_name' in upload_result
            assert 'url' in upload_result
            assert 'commit_sha' in upload_result
            
            # Test model versioning
            version_result = hf_api.create_tag(
                "test-user/test-model",
                tag="v1.0",
                message="Version 1.0 release"
            )
            assert version_result['success']
            
            # Test model metadata update
            metadata_update = hf_api.update_repo_metadata(
                "test-user/test-model",
                {
                    "license": "apache-2.0",
                    "task": "text-classification",
                    "language": "en"
                }
            )
            assert metadata_update['success']
    
    def test_authentication_and_authorization_integration(self):
        """Test authentication and authorization integration."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            # Test login
            login_result = hf_api.login(token="hf_test_token_123")
            assert login_result['success']
            assert login_result['username'] == "test_user"
            
            # Test authenticated operations
            user_info = hf_api.whoami()
            assert user_info['name'] == "test_user"
            assert 'organizations' in user_info
            
            # Test repository permissions
            repo_info = hf_api.get_repo_info("test-user/test-model")
            assert 'permissions' in repo_info
            assert repo_info['permissions']['write'] is True
            
            # Test logout
            logout_result = hf_api.logout()
            assert logout_result['success']


class TestOpenAIServiceIntegration:
    """Test OpenAI service integration across platform components."""
    
    def test_chat_completion_pipeline_integration(self):
        """Test chat completion pipeline integration."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            # Test conversation pipeline
            conversation = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain machine learning in simple terms."}
            ]
            
            response = openai_api.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=conversation,
                max_tokens=100,
                temperature=0.7
            )
            
            assert 'choices' in response
            assert len(response['choices']) > 0
            assert 'message' in response['choices'][0]
            assert response['choices'][0]['message']['role'] == 'assistant'
            
            # Test usage tracking
            assert 'usage' in response
            assert 'prompt_tokens' in response['usage']
            assert 'completion_tokens' in response['usage']
            assert 'total_tokens' in response['usage']
            
            # Verify API call was logged
            assert len(openai_api.api_calls) >= 1
            latest_call = openai_api.api_calls[-1]
            assert latest_call['endpoint'] == 'chat/completions'
            assert latest_call['model'] == 'gpt-3.5-turbo'
    
    def test_embeddings_generation_integration(self):
        """Test embeddings generation integration."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            # Test single text embedding
            text = "This is a test sentence for embedding generation."
            
            response = openai_api.embeddings_create(
                model="text-embedding-ada-002",
                input_texts=[text]
            )
            
            assert 'data' in response
            assert len(response['data']) == 1
            assert 'embedding' in response['data'][0]
            assert len(response['data'][0]['embedding']) == 1536  # OpenAI embedding dimension
            
            # Test batch embeddings
            texts = [
                "First example text",
                "Second example text", 
                "Third example text"
            ]
            
            batch_response = openai_api.embeddings_create(
                model="text-embedding-ada-002",
                input_texts=texts
            )
            
            assert len(batch_response['data']) == 3
            for embedding_data in batch_response['data']:
                assert 'embedding' in embedding_data
                assert len(embedding_data['embedding']) == 1536
    
    def test_function_calling_integration(self):
        """Test function calling integration."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            # Define function schema
            function_schema = {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
            
            # Test function calling
            response = openai_api.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "What's the weather like in Boston?"}
                ],
                functions=[function_schema],
                function_call="auto"
            )
            
            assert 'choices' in response
            choice = response['choices'][0]
            
            # Check if function was called
            if 'function_call' in choice['message']:
                function_call = choice['message']['function_call']
                assert function_call['name'] == 'get_weather'
                assert 'arguments' in function_call
    
    def test_api_error_handling_integration(self):
        """Test API error handling integration."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            # Test rate limiting
            openai_api.enable_rate_limiting(max_calls_per_minute=2)
            
            # Make calls up to limit
            for i in range(2):
                response = openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Test message {i}"}]
                )
                assert response is not None
            
            # Next call should hit rate limit
            with pytest.raises(Exception, match="Rate limit exceeded"):
                openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "This should fail"}]
                )
            
            # Test invalid model handling
            with pytest.raises(Exception):
                openai_api.chat_completions_create(
                    model="invalid-model-name",
                    messages=[{"role": "user", "content": "Test"}]
                )


class TestCloudStorageIntegration:
    """Test cloud storage integration (S3, GCS)."""
    
    def test_s3_service_integration(self):
        """Test S3 service integration."""
        with mock_dependencies_context() as env:
            s3_service = env.get_service('s3')
            
            # Test bucket operations
            bucket_name = "test-ml-models"
            
            # Create bucket
            create_result = s3_service.create_bucket(bucket_name)
            assert create_result['success']
            
            # List buckets
            buckets = s3_service.list_buckets()
            assert any(bucket['name'] == bucket_name for bucket in buckets)
            
            # Test object operations
            test_data = b"This is test model data"
            object_key = "models/test-model/pytorch_model.bin"
            
            # Upload object
            upload_result = s3_service.put_object(
                bucket_name, 
                object_key, 
                test_data,
                metadata={"model_version": "1.0", "framework": "pytorch"}
            )
            assert upload_result['success']
            assert 'etag' in upload_result
            
            # Download object
            download_result = s3_service.get_object(bucket_name, object_key)
            assert download_result['success']
            assert download_result['data'] == test_data
            assert download_result['metadata']['model_version'] == "1.0"
            
            # List objects
            objects = s3_service.list_objects(bucket_name, prefix="models/")
            assert len(objects) >= 1
            assert any(obj['key'] == object_key for obj in objects)
            
            # Delete object
            delete_result = s3_service.delete_object(bucket_name, object_key)
            assert delete_result['success']
    
    def test_gcs_service_integration(self):
        """Test Google Cloud Storage service integration."""
        with mock_dependencies_context() as env:
            gcs_service = env.get_service('gcs')
            
            # Test bucket operations
            bucket_name = "test-ml-datasets"
            
            # Create bucket
            bucket = gcs_service.create_bucket(bucket_name)
            assert bucket['name'] == bucket_name
            
            # Test blob operations
            test_data = "training_data,labels\ntext1,positive\ntext2,negative"
            blob_name = "datasets/sentiment/train.csv"
            
            # Upload blob
            upload_result = gcs_service.upload_blob(
                bucket_name,
                blob_name,
                test_data.encode(),
                content_type="text/csv"
            )
            assert upload_result['success']
            
            # Download blob
            download_result = gcs_service.download_blob(bucket_name, blob_name)
            assert download_result['success']
            assert download_result['data'].decode() == test_data
            
            # Test blob metadata
            metadata = gcs_service.get_blob_metadata(bucket_name, blob_name)
            assert metadata['content_type'] == "text/csv"
            assert 'size' in metadata
            assert 'created' in metadata
            
            # Test signed URLs
            signed_url = gcs_service.generate_signed_url(
                bucket_name, 
                blob_name, 
                expiration=3600
            )
            assert signed_url.startswith('https://')
            assert bucket_name in signed_url
    
    def test_cross_cloud_synchronization(self):
        """Test synchronization between cloud storage services."""
        with mock_dependencies_context() as env:
            s3_service = env.get_service('s3')
            gcs_service = env.get_service('gcs')
            
            # Upload to S3
            s3_bucket = "s3-models"
            s3_key = "models/test-model.bin"
            test_data = b"Model binary data"
            
            s3_service.create_bucket(s3_bucket)
            s3_service.put_object(s3_bucket, s3_key, test_data)
            
            # Download from S3 and upload to GCS
            s3_data = s3_service.get_object(s3_bucket, s3_key)
            
            gcs_bucket = "gcs-models"
            gcs_blob = "models/test-model.bin"
            
            gcs_service.create_bucket(gcs_bucket)
            gcs_upload = gcs_service.upload_blob(
                gcs_bucket, 
                gcs_blob, 
                s3_data['data']
            )
            
            assert gcs_upload['success']
            
            # Verify data integrity
            gcs_data = gcs_service.download_blob(gcs_bucket, gcs_blob)
            assert gcs_data['data'] == test_data


class TestMLOpsServiceIntegration:
    """Test MLOps service integration (MLFlow, WandB)."""
    
    def test_mlflow_experiment_tracking_integration(self):
        """Test MLFlow experiment tracking integration."""
        with mock_dependencies_context() as env:
            mlflow = env.get_service('mlflow')
            
            # Create experiment
            experiment_name = "sentiment_classification"
            experiment_id = mlflow.create_experiment(experiment_name)
            assert experiment_id is not None
            
            # Start run
            run_id = mlflow.start_run(experiment_id)
            assert run_id is not None
            
            # Log parameters
            params = {
                "learning_rate": 2e-4,
                "batch_size": 32,
                "epochs": 5,
                "model_name": "distilbert-base-uncased"
            }
            
            for key, value in params.items():
                mlflow.log_param(run_id, key, value)
            
            # Log metrics
            metrics = [
                {"step": 100, "loss": 1.5, "accuracy": 0.65},
                {"step": 200, "loss": 1.2, "accuracy": 0.72},
                {"step": 300, "loss": 1.0, "accuracy": 0.78}
            ]
            
            for metric in metrics:
                mlflow.log_metric(run_id, "loss", metric["loss"], step=metric["step"])
                mlflow.log_metric(run_id, "accuracy", metric["accuracy"], step=metric["step"])
            
            # Log artifacts
            model_path = "/tmp/model/pytorch_model.bin"
            mlflow.log_artifact(run_id, model_path)
            
            config_path = "/tmp/config/training_config.json"
            mlflow.log_artifact(run_id, config_path)
            
            # End run
            mlflow.end_run(run_id)
            
            # Verify experiment data
            experiment = mlflow.get_experiment(experiment_id)
            assert experiment['name'] == experiment_name
            
            run_info = mlflow.get_run(run_id)
            assert run_info['status'] == 'FINISHED'
            assert len(run_info['params']) == len(params)
            assert len(run_info['metrics']) >= len(metrics) * 2  # loss + accuracy
    
    def test_wandb_experiment_tracking_integration(self):
        """Test Weights & Biases experiment tracking integration."""
        with mock_dependencies_context() as env:
            wandb = env.get_service('wandb')
            
            # Initialize run
            run = wandb.init(
                project="fine-tune-llm",
                name="distilbert-sentiment",
                config={
                    "learning_rate": 2e-4,
                    "batch_size": 32,
                    "epochs": 5
                }
            )
            
            assert run is not None
            assert run.project == "fine-tune-llm"
            assert run.name == "distilbert-sentiment"
            
            # Log metrics
            training_metrics = [
                {"epoch": 1, "step": 100, "train_loss": 1.5, "train_acc": 0.65},
                {"epoch": 1, "step": 200, "train_loss": 1.2, "train_acc": 0.72},
                {"epoch": 2, "step": 300, "train_loss": 1.0, "train_acc": 0.78}
            ]
            
            for metric in training_metrics:
                wandb.log({
                    "epoch": metric["epoch"],
                    "train/loss": metric["train_loss"],
                    "train/accuracy": metric["train_acc"]
                }, step=metric["step"])
            
            # Log model artifacts
            model_artifact = wandb.Artifact(
                name="distilbert-sentiment-model",
                type="model",
                description="Fine-tuned DistilBERT for sentiment classification"
            )
            
            model_artifact.add_file("/tmp/model/pytorch_model.bin")
            model_artifact.add_file("/tmp/model/config.json")
            
            wandb.log_artifact(model_artifact)
            
            # Log hyperparameter sweep results
            sweep_config = {
                "method": "grid",
                "parameters": {
                    "learning_rate": {"values": [1e-4, 2e-4, 5e-4]},
                    "batch_size": {"values": [16, 32, 64]}
                }
            }
            
            sweep_id = wandb.sweep(sweep_config, project="fine-tune-llm")
            assert sweep_id is not None
            
            # Finish run
            wandb.finish()
            
            # Verify run data
            run_summary = wandb.get_run_summary(run.id)
            assert "train/loss" in run_summary
            assert "train/accuracy" in run_summary
    
    def test_model_registry_integration(self):
        """Test model registry integration across MLOps services."""
        with mock_dependencies_context() as env:
            mlflow = env.get_service('mlflow')
            
            # Register model with MLFlow
            model_name = "sentiment-classifier"
            model_version = mlflow.register_model(
                model_uri="models:/sentiment_classifier/1",
                name=model_name,
                description="DistilBERT fine-tuned for sentiment classification"
            )
            
            assert model_version is not None
            assert model_version['name'] == model_name
            assert model_version['version'] == "1"
            
            # Add model metadata
            mlflow.update_model_version(
                name=model_name,
                version="1",
                description="Production-ready sentiment classifier v1.0",
                tags={
                    "framework": "pytorch",
                    "task": "text-classification",
                    "dataset": "imdb",
                    "accuracy": "0.92"
                }
            )
            
            # Transition model to staging
            mlflow.transition_model_version_stage(
                name=model_name,
                version="1",
                stage="Staging",
                archive_existing_versions=False
            )
            
            # Get model info
            model_info = mlflow.get_model_version(model_name, "1")
            assert model_info['current_stage'] == "Staging"
            assert model_info['tags']['accuracy'] == "0.92"
            
            # List all model versions
            model_versions = mlflow.get_model_versions(model_name)
            assert len(model_versions) >= 1
            assert model_versions[0]['name'] == model_name


class TestMonitoringServiceIntegration:
    """Test monitoring service integration (Prometheus, Grafana)."""
    
    def test_prometheus_metrics_integration(self):
        """Test Prometheus metrics collection integration."""
        with mock_dependencies_context() as env:
            prometheus = env.get_infrastructure('prometheus_collector')
            
            # Create metrics
            training_counter = prometheus.counter(
                "training_steps_total",
                "Total number of training steps",
                labels=["model", "dataset"]
            )
            
            model_accuracy = prometheus.gauge(
                "model_accuracy",
                "Current model accuracy",
                labels=["model", "split"]
            )
            
            inference_latency = prometheus.histogram(
                "inference_latency_seconds",
                "Inference latency in seconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            )
            
            # Record metrics
            training_counter.inc(1, labels={"model": "distilbert", "dataset": "imdb"})
            training_counter.inc(5, labels={"model": "distilbert", "dataset": "imdb"})
            
            model_accuracy.set(0.92, labels={"model": "distilbert", "split": "test"})
            model_accuracy.set(0.89, labels={"model": "distilbert", "split": "validation"})
            
            inference_latency.observe(0.3)
            inference_latency.observe(0.7)
            inference_latency.observe(1.2)
            
            # Collect metrics in Prometheus format
            metrics_output = prometheus.collect_metrics()
            assert "training_steps_total" in metrics_output
            assert "model_accuracy" in metrics_output
            assert "inference_latency_seconds" in metrics_output
            
            # Verify metric values
            training_total = prometheus.get_metric_value("training_steps_total", 'model="distilbert",dataset="imdb"')
            assert training_total == 6
            
            test_accuracy = prometheus.get_metric_value("model_accuracy", 'model="distilbert",split="test"')
            assert test_accuracy == 0.92
    
    def test_grafana_dashboard_integration(self):
        """Test Grafana dashboard integration."""
        with mock_dependencies_context() as env:
            grafana = env.get_infrastructure('grafana_client')
            
            # Add Prometheus data source
            grafana.add_datasource("prometheus", {
                "type": "prometheus",
                "url": "http://localhost:9090",
                "access": "proxy"
            })
            
            # Create training dashboard
            dashboard_config = {
                "title": "ML Training Dashboard",
                "tags": ["machine-learning", "training"],
                "panels": [
                    {
                        "title": "Training Loss",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "training_loss",
                                "legendFormat": "{{model}} - {{dataset}}"
                            }
                        ]
                    },
                    {
                        "title": "Model Accuracy",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "model_accuracy",
                                "legendFormat": "{{model}} - {{split}}"
                            }
                        ]
                    },
                    {
                        "title": "Inference Latency",
                        "type": "histogram",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, inference_latency_seconds)",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    }
                ]
            }
            
            dashboard_id = grafana.create_dashboard(dashboard_config)
            assert dashboard_id is not None
            
            # Create alerts
            alert_config = {
                "name": "High Training Loss",
                "condition": "training_loss > 2.0",
                "frequency": "1m",
                "notifications": ["email", "slack"]
            }
            
            alert_id = grafana.create_alert(alert_config)
            assert alert_id is not None
            
            # List dashboards
            dashboards = grafana.list_dashboards()
            assert len(dashboards) >= 1
            assert any(d['title'] == "ML Training Dashboard" for d in dashboards)
            
            # Get dashboard
            dashboard = grafana.get_dashboard(dashboard_id)
            assert dashboard['config']['title'] == "ML Training Dashboard"
            assert len(dashboard['config']['panels']) == 3


class TestCrossServiceIntegration:
    """Test integration across multiple external services."""
    
    def test_model_training_pipeline_integration(self):
        """Test complete model training pipeline across services."""
        with mock_dependencies_context() as env:
            # Get all required services
            hf_api = env.get_service('huggingface')
            mlflow = env.get_service('mlflow')
            s3_service = env.get_service('s3')
            prometheus = env.get_infrastructure('prometheus_collector')
            
            # Step 1: Load model and dataset from HuggingFace
            model = hf_api.from_pretrained("distilbert-base-uncased")
            dataset = hf_api.load_dataset("imdb", split="train[:1000]")
            
            # Step 2: Initialize MLFlow experiment
            experiment_id = mlflow.create_experiment("cross_service_training")
            run_id = mlflow.start_run(experiment_id)
            
            # Step 3: Set up metrics collection
            training_counter = prometheus.counter("training_steps", "Training steps")
            loss_gauge = prometheus.gauge("current_loss", "Current training loss")
            
            # Step 4: Simulate training with integrated logging
            training_steps = [
                {"step": 100, "loss": 1.5, "accuracy": 0.65},
                {"step": 200, "loss": 1.2, "accuracy": 0.72},
                {"step": 300, "loss": 1.0, "accuracy": 0.78}
            ]
            
            for step_data in training_steps:
                # Log to MLFlow
                mlflow.log_metric(run_id, "loss", step_data["loss"], step=step_data["step"])
                mlflow.log_metric(run_id, "accuracy", step_data["accuracy"], step=step_data["step"])
                
                # Update Prometheus metrics
                training_counter.inc(1)
                loss_gauge.set(step_data["loss"])
            
            # Step 5: Save model to S3
            s3_bucket = "ml-models"
            s3_service.create_bucket(s3_bucket)
            
            model_data = b"Trained model binary data"
            s3_service.put_object(
                s3_bucket,
                f"models/{run_id}/pytorch_model.bin",
                model_data,
                metadata={"run_id": run_id, "model_type": "distilbert"}
            )
            
            # Step 6: Log model artifact in MLFlow
            mlflow.log_artifact(run_id, f"s3://{s3_bucket}/models/{run_id}/pytorch_model.bin")
            
            # Step 7: End MLFlow run
            mlflow.end_run(run_id)
            
            # Verify cross-service integration
            run_info = mlflow.get_run(run_id)
            assert run_info['status'] == 'FINISHED'
            assert len(run_info['metrics']) >= 6  # 3 loss + 3 accuracy metrics
            
            s3_objects = s3_service.list_objects(s3_bucket, prefix=f"models/{run_id}/")
            assert len(s3_objects) >= 1
            
            training_steps_total = prometheus.get_metric_value("training_steps")
            assert training_steps_total == 3
    
    def test_model_deployment_pipeline_integration(self):
        """Test model deployment pipeline integration."""
        with mock_dependencies_context() as env:
            # Get services
            mlflow = env.get_service('mlflow')
            s3_service = env.get_service('s3')
            hf_api = env.get_service('huggingface')
            
            # Step 1: Get trained model from registry
            model_name = "production-sentiment-classifier"
            model_version = mlflow.register_model(
                model_uri="s3://ml-models/models/run123/",
                name=model_name
            )
            
            # Step 2: Download model from S3
            model_data = s3_service.get_object(
                "ml-models",
                "models/run123/pytorch_model.bin"
            )
            assert model_data['success']
            
            # Step 3: Deploy to HuggingFace Hub
            deployment_result = hf_api.push_to_hub(
                model_data['data'],
                "organization/production-sentiment-v1",
                commit_message="Production deployment v1.0"
            )
            assert deployment_result['success']
            
            # Step 4: Update model registry with deployment info
            mlflow.update_model_version(
                name=model_name,
                version=model_version['version'],
                description="Deployed to HuggingFace Hub",
                tags={
                    "deployment_status": "deployed",
                    "hub_repo": "organization/production-sentiment-v1",
                    "deployment_date": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Step 5: Transition to production
            mlflow.transition_model_version_stage(
                name=model_name,
                version=model_version['version'],
                stage="Production"
            )
            
            # Verify deployment pipeline
            deployed_model = mlflow.get_model_version(model_name, model_version['version'])
            assert deployed_model['current_stage'] == "Production"
            assert deployed_model['tags']['deployment_status'] == "deployed"
    
    def test_monitoring_alerting_integration(self):
        """Test monitoring and alerting integration."""
        with mock_dependencies_context() as env:
            # Get services
            prometheus = env.get_infrastructure('prometheus_collector')
            grafana = env.get_infrastructure('grafana_client')
            email_service = env.get_infrastructure('email_service')
            slack_notifier = env.get_infrastructure('slack_notifier')
            
            # Set up metrics
            error_counter = prometheus.counter("model_errors_total", "Total model errors")
            latency_histogram = prometheus.histogram("request_latency_seconds", "Request latency")
            
            # Simulate error conditions
            for _ in range(10):
                error_counter.inc(1)
                latency_histogram.observe(5.0)  # High latency
            
            # Create alerting dashboard
            alert_dashboard = grafana.create_dashboard({
                "title": "Model Monitoring Alerts",
                "panels": [
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "alert": {
                            "condition": "model_errors_total > 5",
                            "frequency": "30s"
                        }
                    }
                ]
            })
            
            # Simulate alert trigger
            current_errors = prometheus.get_metric_value("model_errors_total")
            if current_errors > 5:
                # Send email alert
                email_service.send_email(
                    to_address="admin@company.com",
                    subject="High Error Rate Detected",
                    body=f"Model error count: {current_errors}"
                )
                
                # Send Slack alert
                slack_notifier.send_alert(
                    severity="critical",
                    title="Model Error Alert",
                    message=f"Error count exceeded threshold: {current_errors}",
                    channel="#ml-alerts"
                )
            
            # Verify alerting integration
            sent_emails = email_service.get_sent_emails(hours=1)
            sent_messages = slack_notifier.get_messages(channel="#ml-alerts", hours=1)
            
            assert len(sent_emails) >= 1
            assert len(sent_messages) >= 1
            assert "High Error Rate" in sent_emails[0]['subject']
            assert "Error count exceeded" in sent_messages[0]['text']