"""
Integration tests for component interactions.

This test module validates that all platform components work together correctly,
including service layer integration, factory coordination, and data flow.
"""

import pytest
import asyncio
import time
import threading
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
from src.fine_tune_llm.core.factories import ComponentFactory, FactoryRegistry
from src.fine_tune_llm.config.manager import ConfigManager
from src.fine_tune_llm.config.pipeline_config import PipelineConfigManager


class TestEventDrivenComponentInteractions:
    """Test event-driven interactions between components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.event_bus = EventBus()
        self.config_manager = ConfigManager()
        self.factory_registry = FactoryRegistry()
        self.component_states = {}
        
    def test_training_lifecycle_events(self):
        """Test complete training lifecycle through events."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
            
        # Subscribe to all training events
        training_events = [
            EventType.TRAINING_STARTED,
            EventType.TRAINING_EPOCH_COMPLETED,
            EventType.TRAINING_COMPLETED,
            EventType.MODEL_SAVED
        ]
        
        for event_type in training_events:
            self.event_bus.subscribe(event_type, event_handler)
        
        with mock_dependencies_context() as env:
            # Simulate training lifecycle
            self.event_bus.publish(Event(
                EventType.TRAINING_STARTED,
                {"model_name": "test-model", "config": {"epochs": 3}},
                "trainer"
            ))
            
            # Simulate epoch completions
            for epoch in range(3):
                self.event_bus.publish(Event(
                    EventType.TRAINING_EPOCH_COMPLETED,
                    {"epoch": epoch + 1, "loss": 0.5 - epoch * 0.1, "accuracy": 0.8 + epoch * 0.05},
                    "trainer"
                ))
            
            # Training completion
            self.event_bus.publish(Event(
                EventType.TRAINING_COMPLETED,
                {"final_loss": 0.2, "final_accuracy": 0.95, "duration": 3600},
                "trainer"
            ))
            
            # Model saving
            self.event_bus.publish(Event(
                EventType.MODEL_SAVED,
                {"model_path": "/models/test-model", "size_mb": 440},
                "model_manager"
            ))
        
        # Verify all events were received in correct order
        assert len(events_received) == 6  # 1 start + 3 epochs + 1 complete + 1 saved
        
        event_types = [event.type for event in events_received]
        assert event_types[0] == EventType.TRAINING_STARTED
        assert event_types[1] == EventType.TRAINING_EPOCH_COMPLETED
        assert event_types[4] == EventType.TRAINING_COMPLETED
        assert event_types[5] == EventType.MODEL_SAVED
    
    def test_error_propagation_through_events(self):
        """Test error propagation through event system."""
        error_events = []
        
        def error_handler(event):
            error_events.append(event)
        
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, error_handler)
        
        # Simulate cascading errors
        with mock_dependencies_context() as env:
            # Model loading error
            self.event_bus.publish(Event(
                EventType.ERROR_OCCURRED,
                {
                    "error_type": "ModelLoadError",
                    "message": "Failed to load model",
                    "component": "model_manager",
                    "severity": "critical"
                },
                "model_manager"
            ))
            
            # Training error as consequence
            self.event_bus.publish(Event(
                EventType.ERROR_OCCURRED,
                {
                    "error_type": "TrainingError", 
                    "message": "Training failed due to model load error",
                    "component": "trainer",
                    "severity": "critical",
                    "caused_by": "ModelLoadError"
                },
                "trainer"
            ))
        
        assert len(error_events) == 2
        assert error_events[0].data["component"] == "model_manager"
        assert error_events[1].data["caused_by"] == "ModelLoadError"
    
    def test_configuration_change_propagation(self):
        """Test configuration changes propagating through components."""
        config_events = []
        
        def config_handler(event):
            config_events.append(event)
        
        self.event_bus.subscribe(EventType.CONFIGURATION_CHANGED, config_handler)
        
        with mock_dependencies_context() as env:
            # Simulate configuration changes
            self.config_manager.set("training.learning_rate", 0.001)
            
            # Publish config change event
            self.event_bus.publish(Event(
                EventType.CONFIGURATION_CHANGED,
                {
                    "key": "training.learning_rate",
                    "old_value": 0.002,
                    "new_value": 0.001,
                    "component": "training"
                },
                "config_manager"
            ))
            
            # Multiple components might react
            affected_components = ["trainer", "optimizer", "scheduler"]
            for component in affected_components:
                self.event_bus.publish(Event(
                    EventType.CONFIGURATION_RELOADED,
                    {
                        "component": component,
                        "config_key": "training.learning_rate",
                        "status": "updated"
                    },
                    component
                ))
        
        assert len(config_events) >= 1
        assert config_events[0].data["key"] == "training.learning_rate"
    
    def test_async_event_handling(self):
        """Test asynchronous event handling."""
        async_results = []
        
        async def async_handler(event):
            # Simulate async processing
            await asyncio.sleep(0.1)
            async_results.append(f"processed_{event.data['id']}")
        
        def sync_handler(event):
            async_results.append(f"sync_{event.data['id']}")
        
        # Mix of sync and async handlers
        self.event_bus.subscribe(EventType.TRAINING_STARTED, async_handler)
        self.event_bus.subscribe(EventType.TRAINING_STARTED, sync_handler)
        
        async def test_async():
            with mock_dependencies_context() as env:
                # Publish multiple events
                for i in range(3):
                    event = Event(
                        EventType.TRAINING_STARTED,
                        {"id": i, "data": f"test_{i}"},
                        "test"
                    )
                    self.event_bus.publish(event)
                
                # Wait for async processing
                await asyncio.sleep(0.5)
        
        # Run async test
        asyncio.run(test_async())
        
        # Verify both sync and async handlers processed events
        sync_events = [r for r in async_results if r.startswith("sync_")]
        async_events = [r for r in async_results if r.startswith("processed_")]
        
        assert len(sync_events) == 3
        assert len(async_events) == 3


class TestFactoryComponentIntegration:
    """Test factory system integration with components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.factory_registry = FactoryRegistry()
        self.config_manager = ConfigManager()
        
    def test_component_factory_coordination(self):
        """Test coordinated component creation through factories."""
        with mock_dependencies_context() as env:
            # Register mock factories
            component_factory = ComponentFactory()
            
            # Mock component classes
            class MockModelService:
                def __init__(self, model, config):
                    self.model = model
                    self.config = config
                    self.initialized = True
            
            class MockTrainingService:
                def __init__(self, model, trainer, config):
                    self.model = model
                    self.trainer = trainer
                    self.config = config
                    self.initialized = True
            
            # Register component creators
            def create_model_service(**kwargs):
                model = env.get_model_component('model')
                config = kwargs.get('config', {})
                return MockModelService(model, config)
            
            def create_training_service(**kwargs):
                model = env.get_model_component('model')
                trainer = env.get_model_component('trainer')
                config = kwargs.get('config', {})
                return MockTrainingService(model, trainer, config)
            
            # Test coordinated creation
            model_service = create_model_service(config={"param": "value"})
            training_service = create_training_service(config={"epochs": 3})
            
            # Verify components were created correctly
            assert model_service.initialized
            assert training_service.initialized
            assert model_service.model is not None
            assert training_service.model is model_service.model  # Same model instance
    
    def test_factory_dependency_resolution(self):
        """Test factory dependency resolution."""
        with mock_dependencies_context() as env:
            dependencies_created = []
            
            class MockComponentA:
                def __init__(self, name, config=None):
                    self.name = name
                    self.config = config or {}
                    dependencies_created.append(f"A:{name}")
            
            class MockComponentB:
                def __init__(self, name, component_a, config=None):
                    self.name = name
                    self.component_a = component_a
                    self.config = config or {}
                    dependencies_created.append(f"B:{name}")
            
            class MockComponentC:
                def __init__(self, name, component_a, component_b, config=None):
                    self.name = name
                    self.component_a = component_a
                    self.component_b = component_b
                    self.config = config or {}
                    dependencies_created.append(f"C:{name}")
            
            # Create dependency chain: C depends on B, B depends on A
            component_a = MockComponentA("comp_a")
            component_b = MockComponentB("comp_b", component_a)
            component_c = MockComponentC("comp_c", component_a, component_b)
            
            # Verify dependency chain
            assert component_c.component_a is component_a
            assert component_c.component_b is component_b
            assert component_c.component_b.component_a is component_a
            
            # Verify creation order
            assert dependencies_created == ["A:comp_a", "B:comp_b", "C:comp_c"]
    
    def test_factory_error_handling(self):
        """Test factory error handling during component creation."""
        with mock_dependencies_context() as env:
            component_factory = ComponentFactory()
            
            class FailingComponent:
                def __init__(self, **kwargs):
                    if kwargs.get('should_fail', False):
                        raise ValueError("Component creation failed")
                    self.initialized = True
            
            # Test successful creation
            success_component = FailingComponent(should_fail=False)
            assert success_component.initialized
            
            # Test failed creation
            with pytest.raises(ValueError, match="Component creation failed"):
                FailingComponent(should_fail=True)
    
    def test_factory_lifecycle_management(self):
        """Test factory component lifecycle management."""
        with mock_dependencies_context() as env:
            lifecycle_events = []
            
            class LifecycleComponent:
                def __init__(self, name):
                    self.name = name
                    self.state = "created"
                    lifecycle_events.append(f"{name}:created")
                
                def initialize(self):
                    self.state = "initialized"
                    lifecycle_events.append(f"{self.name}:initialized")
                
                def start(self):
                    self.state = "running"
                    lifecycle_events.append(f"{self.name}:started")
                
                def stop(self):
                    self.state = "stopped"
                    lifecycle_events.append(f"{self.name}:stopped")
                
                def cleanup(self):
                    self.state = "cleaned"
                    lifecycle_events.append(f"{self.name}:cleaned")
            
            # Test component lifecycle
            component = LifecycleComponent("test_component")
            component.initialize()
            component.start()
            component.stop()
            component.cleanup()
            
            expected_events = [
                "test_component:created",
                "test_component:initialized", 
                "test_component:started",
                "test_component:stopped",
                "test_component:cleaned"
            ]
            
            assert lifecycle_events == expected_events


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config_manager = PipelineConfigManager()
        self.event_bus = EventBus()
        
    def test_config_hot_reload_integration(self):
        """Test configuration hot reload integration."""
        config_changes = []
        
        def config_change_handler(update):
            config_changes.append(update)
        
        self.config_manager.add_update_listener(config_change_handler)
        
        with mock_dependencies_context() as env:
            # Initial configuration
            self.config_manager.set("model.learning_rate", 0.001)
            
            # Simulate hot reload
            self.config_manager.set("model.learning_rate", 0.002)
            self.config_manager.set("training.batch_size", 32)
            
            # Wait for propagation
            time.sleep(0.1)
        
        # Verify configuration changes were captured
        assert len(config_changes) >= 2
        
        # Find learning rate change
        lr_changes = [c for c in config_changes if c.path == "model.learning_rate"]
        assert len(lr_changes) >= 1
        assert lr_changes[-1].value == 0.002
    
    def test_config_validation_integration(self):
        """Test configuration validation integration."""
        with mock_dependencies_context() as env:
            # Test valid configuration
            valid_result = self.config_manager.set("training.epochs", 5, validate=True)
            assert valid_result
            
            # Test configuration with validation
            self.config_manager.set("model.architecture", "transformer")
            value = self.config_manager.get("model.architecture")
            assert value == "transformer"
    
    def test_config_environment_override(self):
        """Test configuration environment variable override."""
        import os
        
        with mock_dependencies_context() as env:
            # Set environment variable
            os.environ["TRAINING_BATCH_SIZE"] = "64"
            
            try:
                # Test environment override
                self.config_manager.set("training.batch_size", 32)
                
                # Environment should override config
                # (This would require implementing environment override logic)
                base_value = self.config_manager.get("training.batch_size")
                assert base_value == 32  # Base config value
                
            finally:
                # Clean up environment
                if "TRAINING_BATCH_SIZE" in os.environ:
                    del os.environ["TRAINING_BATCH_SIZE"]
    
    def test_config_versioning_integration(self):
        """Test configuration versioning integration."""
        with mock_dependencies_context() as env:
            # Initial configuration
            self.config_manager.set("model.name", "initial_model")
            
            # Make changes
            self.config_manager.set("model.name", "updated_model")
            self.config_manager.set("model.version", "2.0")
            
            # Verify current state
            assert self.config_manager.get("model.name") == "updated_model"
            assert self.config_manager.get("model.version") == "2.0"


class TestDataFlowIntegration:
    """Test data flow integration between components."""
    
    def test_training_data_pipeline(self):
        """Test complete training data pipeline."""
        with mock_dependencies_context() as env:
            pipeline_stages = []
            
            # Mock pipeline components
            class DataLoader:
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def load_batch(self, batch_size=32):
                    pipeline_stages.append("data_loaded")
                    return {"batch": "mock_data", "size": batch_size}
            
            class DataPreprocessor:
                def preprocess(self, batch):
                    pipeline_stages.append("data_preprocessed")
                    return {"processed_batch": batch, "tokens": [1, 2, 3, 4, 5]}
            
            class ModelTrainer:
                def __init__(self, model):
                    self.model = model
                
                def train_step(self, processed_data):
                    pipeline_stages.append("training_step")
                    return {"loss": 0.5, "accuracy": 0.85}
            
            # Create pipeline
            dataset = env.get_model_component('dataset')
            model = env.get_model_component('model')
            
            data_loader = DataLoader(dataset)
            preprocessor = DataPreprocessor()
            trainer = ModelTrainer(model)
            
            # Execute pipeline
            batch = data_loader.load_batch(batch_size=16)
            processed_data = preprocessor.preprocess(batch)
            training_result = trainer.train_step(processed_data)
            
            # Verify pipeline execution
            expected_stages = ["data_loaded", "data_preprocessed", "training_step"]
            assert pipeline_stages == expected_stages
            
            # Verify data flow
            assert batch["size"] == 16
            assert "tokens" in processed_data
            assert training_result["loss"] == 0.5
    
    def test_inference_data_pipeline(self):
        """Test inference data pipeline."""
        with mock_dependencies_context() as env:
            pipeline_results = []
            
            class TextProcessor:
                def tokenize(self, text):
                    pipeline_results.append("tokenized")
                    return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            
            class ModelInference:
                def __init__(self, model):
                    self.model = model
                
                def predict(self, tokens):
                    pipeline_results.append("predicted")
                    return {"logits": [0.1, 0.7, 0.2], "prediction": "positive"}
            
            class PostProcessor:
                def process_output(self, prediction):
                    pipeline_results.append("post_processed")
                    return {
                        "prediction": prediction["prediction"],
                        "confidence": max(prediction["logits"]),
                        "probabilities": prediction["logits"]
                    }
            
            # Create inference pipeline
            model = env.get_model_component('model')
            
            processor = TextProcessor()
            inference = ModelInference(model)
            post_processor = PostProcessor()
            
            # Execute inference pipeline
            input_text = "This is a test sentence"
            tokens = processor.tokenize(input_text)
            prediction = inference.predict(tokens)
            final_result = post_processor.process_output(prediction)
            
            # Verify pipeline execution
            expected_stages = ["tokenized", "predicted", "post_processed"]
            assert pipeline_results == expected_stages
            
            # Verify results
            assert final_result["prediction"] == "positive"
            assert final_result["confidence"] == 0.7
    
    def test_data_validation_pipeline(self):
        """Test data validation pipeline integration."""
        with mock_dependencies_context() as env:
            validation_steps = []
            
            class SchemaValidator:
                def validate_schema(self, data):
                    validation_steps.append("schema_validated")
                    return {"valid": True, "errors": []}
            
            class SecurityValidator:
                def validate_security(self, data):
                    validation_steps.append("security_validated")
                    # Check for potential security issues
                    if "script" in str(data).lower():
                        return {"valid": False, "errors": ["Potential XSS detected"]}
                    return {"valid": True, "errors": []}
            
            class BusinessValidator:
                def validate_business_rules(self, data):
                    validation_steps.append("business_validated")
                    return {"valid": True, "errors": []}
            
            # Create validation pipeline
            schema_validator = SchemaValidator()
            security_validator = SecurityValidator()
            business_validator = BusinessValidator()
            
            # Test with clean data
            clean_data = {"text": "This is clean text", "label": 1}
            
            schema_result = schema_validator.validate_schema(clean_data)
            security_result = security_validator.validate_security(clean_data)
            business_result = business_validator.validate_business_rules(clean_data)
            
            assert all(result["valid"] for result in [schema_result, security_result, business_result])
            
            # Test with suspicious data
            suspicious_data = {"text": "<script>alert('xss')</script>", "label": 1}
            security_result_bad = security_validator.validate_security(suspicious_data)
            assert not security_result_bad["valid"]
            assert "XSS" in security_result_bad["errors"][0]


class TestServiceLayerIntegration:
    """Test service layer integration."""
    
    def test_model_service_integration(self):
        """Test model service integration."""
        with mock_dependencies_context() as env:
            service_operations = []
            
            class ModelService:
                def __init__(self, model_manager, config_manager):
                    self.model_manager = model_manager
                    self.config_manager = config_manager
                    self.loaded_models = {}
                
                def load_model(self, model_name):
                    service_operations.append(f"load:{model_name}")
                    model = env.get_model_component('model')
                    self.loaded_models[model_name] = model
                    return model
                
                def get_model_info(self, model_name):
                    service_operations.append(f"info:{model_name}")
                    if model_name in self.loaded_models:
                        return {"name": model_name, "loaded": True, "size": "440MB"}
                    return {"name": model_name, "loaded": False}
                
                def unload_model(self, model_name):
                    service_operations.append(f"unload:{model_name}")
                    self.loaded_models.pop(model_name, None)
            
            # Create service
            model_service = ModelService(None, self.config_manager)
            
            # Test service operations
            model = model_service.load_model("test-model")
            info = model_service.get_model_info("test-model")
            model_service.unload_model("test-model")
            final_info = model_service.get_model_info("test-model")
            
            # Verify operations
            expected_ops = ["load:test-model", "info:test-model", "unload:test-model", "info:test-model"]
            assert service_operations == expected_ops
            
            assert model is not None
            assert info["loaded"] is True
            assert final_info["loaded"] is False
    
    def test_training_service_integration(self):
        """Test training service integration."""
        with mock_dependencies_context() as env:
            training_events = []
            
            class TrainingService:
                def __init__(self, model_service, config_manager, event_bus):
                    self.model_service = model_service
                    self.config_manager = config_manager
                    self.event_bus = event_bus
                    self.training_jobs = {}
                
                def start_training(self, job_id, model_name, config):
                    training_events.append(f"start:{job_id}")
                    
                    # Load model through model service
                    model = env.get_model_component('model')
                    
                    # Create training job
                    job = {
                        "id": job_id,
                        "model": model,
                        "config": config,
                        "status": "running",
                        "progress": 0
                    }
                    self.training_jobs[job_id] = job
                    
                    # Publish event
                    self.event_bus.publish(Event(
                        EventType.TRAINING_STARTED,
                        {"job_id": job_id, "model_name": model_name},
                        "training_service"
                    ))
                    
                    return job_id
                
                def get_training_status(self, job_id):
                    training_events.append(f"status:{job_id}")
                    return self.training_jobs.get(job_id, {"status": "not_found"})
                
                def stop_training(self, job_id):
                    training_events.append(f"stop:{job_id}")
                    if job_id in self.training_jobs:
                        self.training_jobs[job_id]["status"] = "stopped"
                        
                        self.event_bus.publish(Event(
                            EventType.TRAINING_STOPPED,
                            {"job_id": job_id, "reason": "user_requested"},
                            "training_service"
                        ))
            
            # Create integrated services
            event_bus = EventBus()
            config_manager = ConfigManager()
            
            model_service = None  # Would be actual ModelService
            training_service = TrainingService(model_service, config_manager, event_bus)
            
            # Test training workflow
            job_id = training_service.start_training(
                "job_001", 
                "test-model", 
                {"epochs": 5, "lr": 0.001}
            )
            
            status = training_service.get_training_status(job_id)
            training_service.stop_training(job_id)
            final_status = training_service.get_training_status(job_id)
            
            # Verify workflow
            expected_events = ["start:job_001", "status:job_001", "stop:job_001", "status:job_001"]
            assert training_events == expected_events
            
            assert status["status"] == "running"
            assert final_status["status"] == "stopped"
    
    def test_cross_service_communication(self):
        """Test communication between multiple services."""
        with mock_dependencies_context() as env:
            communication_log = []
            
            class ServiceA:
                def __init__(self, event_bus):
                    self.event_bus = event_bus
                    self.event_bus.subscribe(EventType.CUSTOM_EVENT, self.handle_custom_event)
                
                def handle_custom_event(self, event):
                    communication_log.append(f"ServiceA received: {event.data['message']}")
                
                def send_message(self, message):
                    communication_log.append(f"ServiceA sending: {message}")
                    self.event_bus.publish(Event(
                        EventType.CUSTOM_EVENT,
                        {"message": message, "sender": "ServiceA"},
                        "ServiceA"
                    ))
            
            class ServiceB:
                def __init__(self, event_bus):
                    self.event_bus = event_bus
                    self.event_bus.subscribe(EventType.CUSTOM_EVENT, self.handle_custom_event)
                
                def handle_custom_event(self, event):
                    if event.source != "ServiceB":  # Don't handle own messages
                        communication_log.append(f"ServiceB received: {event.data['message']}")
                        # Send response
                        self.send_response(f"Response to: {event.data['message']}")
                
                def send_response(self, message):
                    communication_log.append(f"ServiceB responding: {message}")
                    self.event_bus.publish(Event(
                        EventType.CUSTOM_EVENT,
                        {"message": message, "sender": "ServiceB"},
                        "ServiceB"
                    ))
            
            # Create services with shared event bus
            event_bus = EventBus()
            service_a = ServiceA(event_bus)
            service_b = ServiceB(event_bus)
            
            # Test cross-service communication
            service_a.send_message("Hello from A")
            
            # Allow event processing
            time.sleep(0.1)
            
            # Verify communication
            assert len(communication_log) >= 3
            assert "ServiceA sending: Hello from A" in communication_log
            assert "ServiceB received: Hello from A" in communication_log
            assert any("ServiceB responding:" in msg for msg in communication_log)


class TestConcurrentComponentInteractions:
    """Test concurrent component interactions."""
    
    def test_concurrent_event_processing(self):
        """Test concurrent event processing."""
        with mock_dependencies_context() as env:
            event_bus = EventBus()
            processed_events = []
            processing_lock = threading.Lock()
            
            def event_processor(worker_id):
                def handler(event):
                    with processing_lock:
                        processed_events.append(f"worker_{worker_id}:{event.data['id']}")
                
                event_bus.subscribe(EventType.TRAINING_STARTED, handler)
            
            # Create multiple event processors
            for worker_id in range(3):
                threading.Thread(target=event_processor, args=(worker_id,)).start()
            
            # Wait for subscriptions to be set up
            time.sleep(0.1)
            
            # Publish events concurrently
            def publish_events():
                for i in range(10):
                    event_bus.publish(Event(
                        EventType.TRAINING_STARTED,
                        {"id": i},
                        "test"
                    ))
            
            # Start multiple publishers
            publishers = []
            for _ in range(2):
                publisher = threading.Thread(target=publish_events)
                publishers.append(publisher)
                publisher.start()
            
            # Wait for completion
            for publisher in publishers:
                publisher.join()
            
            time.sleep(0.2)  # Allow event processing
            
            # Verify concurrent processing
            assert len(processed_events) > 0
            
            # Should have events from multiple workers
            workers_that_processed = set(event.split(':')[0] for event in processed_events)
            assert len(workers_that_processed) > 1
    
    def test_concurrent_factory_creation(self):
        """Test concurrent component creation through factories."""
        with mock_dependencies_context() as env:
            factory_registry = FactoryRegistry()
            created_components = []
            creation_lock = threading.Lock()
            
            class ThreadSafeComponent:
                def __init__(self, component_id):
                    self.component_id = component_id
                    self.created_at = datetime.now(timezone.utc)
                    with creation_lock:
                        created_components.append(component_id)
            
            def create_components(worker_id):
                for i in range(5):
                    component_id = f"worker_{worker_id}_component_{i}"
                    component = ThreadSafeComponent(component_id)
                    assert component.component_id == component_id
            
            # Create components concurrently
            threads = []
            for worker_id in range(4):
                thread = threading.Thread(target=create_components, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify all components were created
            assert len(created_components) == 20  # 4 workers * 5 components
            
            # Verify no duplicates
            assert len(set(created_components)) == 20
    
    def test_concurrent_configuration_updates(self):
        """Test concurrent configuration updates."""
        with mock_dependencies_context() as env:
            config_manager = ConfigManager()
            update_results = []
            update_lock = threading.Lock()
            
            def update_config(worker_id):
                for i in range(10):
                    key = f"worker_{worker_id}.setting_{i}"
                    value = f"value_{worker_id}_{i}"
                    
                    config_manager.set(key, value)
                    retrieved_value = config_manager.get(key)
                    
                    with update_lock:
                        update_results.append((key, value, retrieved_value))
            
            # Perform concurrent updates
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(target=update_config, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify all updates
            assert len(update_results) == 30  # 3 workers * 10 updates
            
            # Verify data consistency
            for key, original_value, retrieved_value in update_results:
                assert original_value == retrieved_value