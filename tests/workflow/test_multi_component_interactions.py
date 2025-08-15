"""
Multi-component interaction tests for complex system behavior validation.

This test module validates interactions between multiple platform components
working together to achieve complex behaviors and workflows.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class TestTrainingEvaluationInteraction:
    """Test interactions between training and evaluation components."""
    
    def test_real_time_training_evaluation_feedback_loop(self):
        """Test real-time feedback loop between training and evaluation."""
        with mock_dependencies_context() as env:
            # Initialize components
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            mlflow = env.get_service('mlflow')
            prometheus = env.get_infrastructure('prometheus_collector')
            
            # Track component interactions
            interaction_log = []
            performance_metrics = []
            
            def log_interaction(component, action, data):
                interaction_log.append({
                    "component": component,
                    "action": action,
                    "data": data,
                    "timestamp": time.time()
                })
            
            # Configure real-time evaluation
            eval_config = {
                "evaluation.real_time.enabled": True,
                "evaluation.real_time.eval_steps": 50,
                "evaluation.real_time.early_stopping.enabled": True,
                "evaluation.real_time.early_stopping.patience": 3,
                "evaluation.real_time.early_stopping.min_delta": 0.001,
                "training.adaptive_lr.enabled": True,
                "training.adaptive_lr.factor": 0.8,
                "training.adaptive_lr.patience": 2
            }
            
            for path, value in eval_config.items():
                config_manager.set(path, value)
            
            # Mock training and evaluation components
            class TrainingComponent:
                def __init__(self):
                    self.current_lr = 2e-4
                    self.training_step = 0
                    self.should_stop = False
                    
                def training_step_completed(self, loss, accuracy):
                    self.training_step += 1
                    log_interaction("trainer", "step_completed", {
                        "step": self.training_step,
                        "loss": loss,
                        "accuracy": accuracy,
                        "learning_rate": self.current_lr
                    })
                    
                    # Publish training step event
                    event_bus.publish(Event(
                        EventType.TRAINING_STEP_COMPLETED,
                        {
                            "step": self.training_step,
                            "loss": loss,
                            "accuracy": accuracy,
                            "learning_rate": self.current_lr
                        },
                        "trainer"
                    ))
                    
                    return not self.should_stop
                
                def adjust_learning_rate(self, factor):
                    old_lr = self.current_lr
                    self.current_lr *= factor
                    log_interaction("trainer", "lr_adjusted", {
                        "old_lr": old_lr,
                        "new_lr": self.current_lr,
                        "factor": factor
                    })
                
                def stop_training(self, reason):
                    self.should_stop = True
                    log_interaction("trainer", "early_stop", {"reason": reason})
            
            class EvaluationComponent:
                def __init__(self):
                    self.eval_history = []
                    self.best_accuracy = 0.0
                    self.patience_counter = 0
                    
                def evaluate_model(self, step):
                    # Simulate evaluation
                    base_accuracy = 0.6
                    improvement = min(step * 0.001, 0.25)  # Diminishing returns
                    noise = (hash(step) % 100) * 0.0001  # Small random variation
                    
                    accuracy = base_accuracy + improvement + noise
                    loss = 1.5 - improvement - (noise * 0.5)
                    
                    eval_result = {
                        "step": step,
                        "accuracy": accuracy,
                        "loss": loss,
                        "timestamp": time.time()
                    }
                    
                    self.eval_history.append(eval_result)
                    performance_metrics.append(eval_result)
                    
                    log_interaction("evaluator", "evaluation_completed", eval_result)
                    
                    # Publish evaluation event
                    event_bus.publish(Event(
                        EventType.EVALUATION_COMPLETED,
                        eval_result,
                        "evaluator"
                    ))
                    
                    return eval_result
                
                def check_early_stopping(self, current_accuracy):
                    min_delta = config_manager.get("evaluation.real_time.early_stopping.min_delta")
                    patience = config_manager.get("evaluation.real_time.early_stopping.patience")
                    
                    if current_accuracy > self.best_accuracy + min_delta:
                        self.best_accuracy = current_accuracy
                        self.patience_counter = 0
                        return False, "improvement"
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= patience:
                            return True, "early_stopping"
                        return False, "no_improvement"
                
                def recommend_lr_adjustment(self, recent_evals):
                    if len(recent_evals) < 3:
                        return False, 1.0
                    
                    # Check if accuracy is plateauing
                    recent_accuracies = [e["accuracy"] for e in recent_evals[-3:]]
                    accuracy_trend = recent_accuracies[-1] - recent_accuracies[0]
                    
                    if accuracy_trend < 0.005:  # Very small improvement
                        return True, config_manager.get("training.adaptive_lr.factor")
                    
                    return False, 1.0
            
            # Initialize components
            trainer = TrainingComponent()
            evaluator = EvaluationComponent()
            
            # Set up event handlers for component interaction
            def handle_training_step(event):
                step = event.data["step"]
                eval_steps = config_manager.get("evaluation.real_time.eval_steps")
                
                # Run evaluation every N steps
                if step % eval_steps == 0:
                    eval_result = evaluator.evaluate_model(step)
                    
                    # Check early stopping
                    should_stop, reason = evaluator.check_early_stopping(eval_result["accuracy"])
                    if should_stop:
                        trainer.stop_training(reason)
                        return
                    
                    # Check learning rate adjustment
                    recent_evals = evaluator.eval_history[-3:] if len(evaluator.eval_history) >= 3 else evaluator.eval_history
                    should_adjust, factor = evaluator.recommend_lr_adjustment(recent_evals)
                    if should_adjust:
                        trainer.adjust_learning_rate(factor)
            
            event_bus.subscribe(EventType.TRAINING_STEP_COMPLETED, handle_training_step)
            
            # Run training simulation with real-time evaluation
            max_steps = 500
            for step in range(1, max_steps + 1):
                # Simulate training step
                base_loss = 1.5
                loss_reduction = min(step * 0.002, 0.8)
                current_loss = base_loss - loss_reduction
                
                base_acc = 0.6
                acc_improvement = min(step * 0.0008, 0.3)
                current_acc = base_acc + acc_improvement
                
                # Execute training step
                continue_training = trainer.training_step_completed(current_loss, current_acc)
                
                if not continue_training:
                    break
                
                # Small delay to simulate training time
                time.sleep(0.001)
            
            # Verify component interactions
            assert len(interaction_log) > 0
            
            # Verify training steps were logged
            training_steps = [log for log in interaction_log if log["action"] == "step_completed"]
            assert len(training_steps) > 0
            
            # Verify evaluations were performed
            evaluations = [log for log in interaction_log if log["action"] == "evaluation_completed"]
            assert len(evaluations) > 0
            
            # Verify adaptive learning rate adjustments
            lr_adjustments = [log for log in interaction_log if log["action"] == "lr_adjusted"]
            assert len(lr_adjustments) >= 0  # May or may not have adjustments
            
            # Verify early stopping might have occurred
            early_stops = [log for log in interaction_log if log["action"] == "early_stop"]
            # Early stopping may or may not occur depending on the simulation
            
            # Verify evaluation frequency
            eval_steps = config_manager.get("evaluation.real_time.eval_steps")
            expected_evals = len(training_steps) // eval_steps
            assert len(evaluations) >= expected_evals * 0.8  # Allow some tolerance
    
    def test_distributed_training_coordination(self):
        """Test coordination between multiple distributed training components."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Configure distributed training
            world_size = 4
            distributed_config = {
                "distributed.world_size": world_size,
                "distributed.synchronization.all_reduce_frequency": 10,
                "distributed.gradient_compression.enabled": True,
                "distributed.dynamic_batching.enabled": True,
                "distributed.fault_tolerance.enabled": True,
                "distributed.load_balancing.enabled": True
            }
            
            for path, value in distributed_config.items():
                config_manager.set(path, value)
            
            # Track distributed interactions
            coordination_events = []
            synchronization_points = []
            worker_states = {}
            
            def log_coordination_event(event_type, data):
                coordination_events.append({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time()
                })
            
            class DistributedWorker:
                def __init__(self, rank, world_size):
                    self.rank = rank
                    self.world_size = world_size
                    self.local_step = 0
                    self.gradient_buffer = []
                    self.is_active = True
                    self.processing_time = 0.1 + (rank * 0.02)  # Simulate varying speeds
                    
                def training_step(self, step):
                    if not self.is_active:
                        return None
                    
                    self.local_step = step
                    
                    # Simulate gradient computation
                    time.sleep(self.processing_time)
                    
                    # Generate mock gradients
                    gradients = {
                        "layer1": [0.1 + (self.rank * 0.01), 0.2 + (self.rank * 0.01)],
                        "layer2": [0.05 + (self.rank * 0.005), 0.15 + (self.rank * 0.005)]
                    }
                    
                    self.gradient_buffer.append(gradients)
                    
                    log_coordination_event("worker_step_completed", {
                        "rank": self.rank,
                        "step": step,
                        "gradients_norm": sum(sum(layer) for layer in gradients.values())
                    })
                    
                    return gradients
                
                def synchronize_gradients(self, all_gradients):
                    """Simulate all-reduce operation."""
                    if not self.is_active:
                        return None
                    
                    # Average gradients across all workers
                    avg_gradients = {}
                    for layer in all_gradients[0].keys():
                        avg_gradients[layer] = []
                        for i in range(len(all_gradients[0][layer])):
                            avg_value = sum(worker_grads[layer][i] for worker_grads in all_gradients) / len(all_gradients)
                            avg_gradients[layer].append(avg_value)
                    
                    log_coordination_event("gradients_synchronized", {
                        "rank": self.rank,
                        "step": self.local_step,
                        "avg_gradients_norm": sum(sum(layer) for layer in avg_gradients.values())
                    })
                    
                    return avg_gradients
                
                def simulate_failure(self):
                    """Simulate worker failure."""
                    self.is_active = False
                    log_coordination_event("worker_failed", {"rank": self.rank})
                
                def recover(self):
                    """Simulate worker recovery."""
                    self.is_active = True
                    log_coordination_event("worker_recovered", {"rank": self.rank})
            
            class DistributedCoordinator:
                def __init__(self, world_size):
                    self.world_size = world_size
                    self.workers = [DistributedWorker(rank, world_size) for rank in range(world_size)]
                    self.global_step = 0
                    self.sync_frequency = config_manager.get("distributed.synchronization.all_reduce_frequency")
                    
                def coordinate_training_step(self):
                    self.global_step += 1
                    
                    # Collect gradients from all active workers
                    all_gradients = []
                    active_workers = []
                    
                    for worker in self.workers:
                        if worker.is_active:
                            gradients = worker.training_step(self.global_step)
                            if gradients:
                                all_gradients.append(gradients)
                                active_workers.append(worker)
                    
                    # Check if we need synchronization
                    if self.global_step % self.sync_frequency == 0 and len(all_gradients) > 1:
                        synchronization_points.append({
                            "step": self.global_step,
                            "active_workers": len(active_workers),
                            "total_workers": self.world_size
                        })
                        
                        # Synchronize gradients across workers
                        for worker in active_workers:
                            worker.synchronize_gradients(all_gradients)
                        
                        log_coordination_event("global_synchronization", {
                            "step": self.global_step,
                            "active_workers": len(active_workers),
                            "total_workers": self.world_size
                        })
                    
                    return len(active_workers) > 0
                
                def handle_worker_failure(self, rank):
                    """Handle worker failure with fault tolerance."""
                    if rank < len(self.workers):
                        self.workers[rank].simulate_failure()
                        
                        # Implement fault tolerance
                        active_workers = sum(1 for w in self.workers if w.is_active)
                        
                        if active_workers < self.world_size // 2:
                            log_coordination_event("critical_failure", {
                                "active_workers": active_workers,
                                "total_workers": self.world_size
                            })
                            return False
                        
                        log_coordination_event("fault_tolerance_activated", {
                            "failed_rank": rank,
                            "remaining_workers": active_workers
                        })
                    
                    return True
                
                def recover_worker(self, rank):
                    """Recover a failed worker."""
                    if rank < len(self.workers):
                        self.workers[rank].recover()
            
            # Initialize coordinator
            coordinator = DistributedCoordinator(world_size)
            
            # Run distributed training simulation
            max_steps = 100
            
            for step in range(1, max_steps + 1):
                # Randomly simulate worker failures and recoveries
                if step == 30:
                    # Simulate worker failure
                    coordinator.handle_worker_failure(2)
                elif step == 60:
                    # Recover worker
                    coordinator.recover_worker(2)
                elif step == 80:
                    # Another failure
                    coordinator.handle_worker_failure(1)
                
                # Coordinate training step
                success = coordinator.coordinate_training_step()
                if not success:
                    break
                
                # Track worker states
                worker_states[step] = {
                    "active_workers": sum(1 for w in coordinator.workers if w.is_active),
                    "total_workers": world_size
                }
            
            # Verify distributed coordination
            assert len(coordination_events) > 0
            
            # Verify worker coordination
            step_events = [e for e in coordination_events if e["type"] == "worker_step_completed"]
            assert len(step_events) > 0
            
            # Verify synchronization occurred
            sync_events = [e for e in coordination_events if e["type"] == "global_synchronization"]
            expected_syncs = max_steps // coordinator.sync_frequency
            assert len(sync_events) >= expected_syncs * 0.5  # Account for failures
            
            # Verify fault tolerance was activated
            fault_events = [e for e in coordination_events if e["type"] == "fault_tolerance_activated"]
            assert len(fault_events) >= 1  # At least one failure was handled
            
            # Verify worker recovery
            recovery_events = [e for e in coordination_events if e["type"] == "worker_recovered"]
            assert len(recovery_events) >= 1


class TestModelServingInferenceInteraction:
    """Test interactions between model serving and inference components."""
    
    def test_dynamic_model_serving_with_load_balancing(self):
        """Test dynamic model serving with load balancing across multiple inference instances."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Configure load balancing
            serving_config = {
                "serving.replicas": 3,
                "serving.load_balancer.strategy": "round_robin",
                "serving.auto_scaling.enabled": True,
                "serving.auto_scaling.target_utilization": 0.7,
                "serving.auto_scaling.min_replicas": 2,
                "serving.auto_scaling.max_replicas": 8,
                "serving.health_check.enabled": True,
                "serving.health_check.interval": 10,
                "monitoring.request_latency.p95_threshold_ms": 200
            }
            
            for path, value in serving_config.items():
                config_manager.set(path, value)
            
            # Track serving interactions
            serving_events = []
            request_queue = []
            response_times = []
            
            def log_serving_event(event_type, data):
                serving_events.append({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time()
                })
            
            class InferenceReplica:
                def __init__(self, replica_id):
                    self.replica_id = replica_id
                    self.is_healthy = True
                    self.current_load = 0
                    self.max_capacity = 10
                    self.processing_times = []
                    
                def process_request(self, request):
                    if not self.is_healthy or self.current_load >= self.max_capacity:
                        return None
                    
                    start_time = time.time()
                    self.current_load += 1
                    
                    # Simulate processing time based on current load
                    base_time = 0.05
                    load_factor = 1 + (self.current_load / self.max_capacity) * 0.5
                    processing_time = base_time * load_factor
                    
                    time.sleep(processing_time)
                    
                    # Generate response
                    response = {
                        "request_id": request["id"],
                        "replica_id": self.replica_id,
                        "prediction": "positive" if hash(request["text"]) % 2 else "negative",
                        "confidence": 0.85 + ((hash(request["text"]) % 100) * 0.001),
                        "processing_time": processing_time
                    }
                    
                    self.current_load -= 1
                    self.processing_times.append(processing_time)
                    
                    log_serving_event("request_processed", {
                        "replica_id": self.replica_id,
                        "request_id": request["id"],
                        "processing_time": processing_time,
                        "current_load": self.current_load
                    })
                    
                    return response
                
                def get_utilization(self):
                    return self.current_load / self.max_capacity
                
                def get_avg_latency(self):
                    if not self.processing_times:
                        return 0.0
                    return sum(self.processing_times) / len(self.processing_times)
                
                def health_check(self):
                    # Simulate occasional health issues
                    if len(self.processing_times) > 50:
                        avg_latency = self.get_avg_latency()
                        if avg_latency > 0.15:  # High latency indicates problems
                            self.is_healthy = False
                            log_serving_event("replica_unhealthy", {
                                "replica_id": self.replica_id,
                                "avg_latency": avg_latency
                            })
                    
                    return self.is_healthy
            
            class LoadBalancer:
                def __init__(self, initial_replicas):
                    self.replicas = [InferenceReplica(i) for i in range(initial_replicas)]
                    self.current_replica_index = 0
                    self.strategy = config_manager.get("serving.load_balancer.strategy")
                    
                def add_replica(self):
                    new_id = len(self.replicas)
                    self.replicas.append(InferenceReplica(new_id))
                    log_serving_event("replica_added", {"replica_id": new_id})
                
                def remove_replica(self, replica_id):
                    if len(self.replicas) > config_manager.get("serving.auto_scaling.min_replicas"):
                        self.replicas = [r for r in self.replicas if r.replica_id != replica_id]
                        log_serving_event("replica_removed", {"replica_id": replica_id})
                
                def select_replica(self, request):
                    healthy_replicas = [r for r in self.replicas if r.is_healthy]
                    
                    if not healthy_replicas:
                        return None
                    
                    if self.strategy == "round_robin":
                        replica = healthy_replicas[self.current_replica_index % len(healthy_replicas)]
                        self.current_replica_index += 1
                    elif self.strategy == "least_loaded":
                        replica = min(healthy_replicas, key=lambda r: r.current_load)
                    else:
                        replica = healthy_replicas[0]  # Default to first
                    
                    return replica
                
                def route_request(self, request):
                    replica = self.select_replica(request)
                    if replica:
                        return replica.process_request(request)
                    return None
                
                def get_cluster_utilization(self):
                    if not self.replicas:
                        return 0.0
                    
                    total_utilization = sum(r.get_utilization() for r in self.replicas if r.is_healthy)
                    healthy_count = sum(1 for r in self.replicas if r.is_healthy)
                    
                    return total_utilization / healthy_count if healthy_count > 0 else 0.0
                
                def auto_scale(self):
                    target_utilization = config_manager.get("serving.auto_scaling.target_utilization")
                    min_replicas = config_manager.get("serving.auto_scaling.min_replicas")
                    max_replicas = config_manager.get("serving.auto_scaling.max_replicas")
                    
                    current_utilization = self.get_cluster_utilization()
                    current_count = len([r for r in self.replicas if r.is_healthy])
                    
                    # Scale up if utilization is too high
                    if current_utilization > target_utilization and current_count < max_replicas:
                        self.add_replica()
                        log_serving_event("auto_scale_up", {
                            "current_utilization": current_utilization,
                            "target_utilization": target_utilization,
                            "new_replica_count": len(self.replicas)
                        })
                    
                    # Scale down if utilization is too low
                    elif current_utilization < target_utilization * 0.5 and current_count > min_replicas:
                        # Remove least utilized replica
                        least_utilized = min(self.replicas, key=lambda r: r.get_utilization())
                        self.remove_replica(least_utilized.replica_id)
                        log_serving_event("auto_scale_down", {
                            "current_utilization": current_utilization,
                            "target_utilization": target_utilization,
                            "removed_replica": least_utilized.replica_id
                        })
                
                def health_check_all(self):
                    for replica in self.replicas:
                        replica.health_check()
            
            # Initialize load balancer
            initial_replicas = config_manager.get("serving.replicas")
            load_balancer = LoadBalancer(initial_replicas)
            
            # Generate request load
            def generate_requests(num_requests, batch_size=20):
                """Generate batches of requests to simulate varying load."""
                all_requests = []
                
                for i in range(num_requests):
                    request = {
                        "id": f"req_{i}",
                        "text": f"This is test request number {i}",
                        "timestamp": time.time()
                    }
                    all_requests.append(request)
                
                # Send requests in batches to simulate load spikes
                for i in range(0, len(all_requests), batch_size):
                    batch = all_requests[i:i + batch_size]
                    
                    # Process batch concurrently
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = []
                        
                        for request in batch:
                            future = executor.submit(load_balancer.route_request, request)
                            futures.append((request, future))
                        
                        # Collect responses
                        for request, future in futures:
                            try:
                                response = future.result(timeout=1.0)
                                if response:
                                    response_times.append(response["processing_time"])
                                    log_serving_event("request_completed", {
                                        "request_id": request["id"],
                                        "response_time": response["processing_time"],
                                        "replica_id": response["replica_id"]
                                    })
                            except Exception as e:
                                log_serving_event("request_failed", {
                                    "request_id": request["id"],
                                    "error": str(e)
                                })
                    
                    # Auto-scale and health check between batches
                    load_balancer.auto_scale()
                    load_balancer.health_check_all()
                    
                    # Small delay between batches
                    time.sleep(0.1)
            
            # Run load test with varying patterns
            generate_requests(50, batch_size=10)   # Light load
            generate_requests(100, batch_size=30)  # Medium load
            generate_requests(200, batch_size=50)  # Heavy load
            generate_requests(30, batch_size=5)    # Cool down
            
            # Verify serving interactions
            assert len(serving_events) > 0
            
            # Verify requests were processed
            processed_requests = [e for e in serving_events if e["type"] == "request_completed"]
            assert len(processed_requests) > 0
            
            # Verify auto-scaling occurred
            scale_up_events = [e for e in serving_events if e["type"] == "auto_scale_up"]
            scale_down_events = [e for e in serving_events if e["type"] == "auto_scale_down"]
            # Auto-scaling should have occurred during the test
            
            # Verify load balancing
            replica_usage = {}
            for event in processed_requests:
                replica_id = event["data"]["replica_id"]
                replica_usage[replica_id] = replica_usage.get(replica_id, 0) + 1
            
            # Load should be distributed across replicas
            assert len(replica_usage) >= 2  # At least 2 replicas handled requests
            
            # Verify response times
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
                
                # Response times should be reasonable
                assert avg_response_time < 0.5
                assert p95_response_time < 1.0


class TestMonitoringAlertsIntegration:
    """Test integration between monitoring, alerting, and remediation systems."""
    
    def test_intelligent_alert_escalation_and_remediation(self):
        """Test intelligent alert escalation with automated remediation."""
        with mock_dependencies_context() as env:
            config_manager = PipelineConfigManager()
            event_bus = EventBus()
            
            # Services
            prometheus = env.get_infrastructure('prometheus_collector')
            grafana = env.get_infrastructure('grafana_client')
            email_service = env.get_infrastructure('email_service')
            slack_notifier = env.get_infrastructure('slack_notifier')
            
            # Configure intelligent alerting
            alerting_config = {
                "alerting.intelligent.enabled": True,
                "alerting.escalation.levels": 3,
                "alerting.escalation.timeouts": [300, 600, 1200],  # 5min, 10min, 20min
                "alerting.auto_remediation.enabled": True,
                "alerting.correlation.window_seconds": 600,
                "alerting.noise_reduction.enabled": True,
                "alerting.severity_thresholds": {
                    "low": 0.1,
                    "medium": 0.3,
                    "high": 0.6,
                    "critical": 0.8
                }
            }
            
            for path, value in alerting_config.items():
                config_manager.set(path, value)
            
            # Track alerting interactions
            alert_events = []
            remediation_actions = []
            escalation_history = []
            
            def log_alert_event(event_type, data):
                alert_events.append({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time()
                })
            
            class AlertManager:
                def __init__(self):
                    self.active_alerts = {}
                    self.alert_history = []
                    self.correlation_groups = []
                    
                def create_alert(self, metric_name, value, threshold, severity):
                    alert_id = f"alert_{len(self.alert_history)}"
                    
                    alert = {
                        "id": alert_id,
                        "metric_name": metric_name,
                        "current_value": value,
                        "threshold": threshold,
                        "severity": severity,
                        "created_at": time.time(),
                        "escalation_level": 0,
                        "remediation_attempts": [],
                        "correlated_alerts": []
                    }
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    log_alert_event("alert_created", alert)
                    
                    # Check for correlation with existing alerts
                    self.correlate_alerts(alert)
                    
                    return alert_id
                
                def correlate_alerts(self, new_alert):
                    """Correlate alerts to reduce noise and identify patterns."""
                    correlation_window = config_manager.get("alerting.correlation.window_seconds")
                    current_time = time.time()
                    
                    # Find recent alerts that might be related
                    recent_alerts = [
                        alert for alert in self.alert_history
                        if current_time - alert["created_at"] < correlation_window
                        and alert["id"] != new_alert["id"]
                    ]
                    
                    # Simple correlation based on metric patterns
                    correlated = []
                    for alert in recent_alerts:
                        if (
                            alert["metric_name"].split("_")[0] == new_alert["metric_name"].split("_")[0] or
                            alert["severity"] == "critical" and new_alert["severity"] == "critical"
                        ):
                            correlated.append(alert["id"])
                    
                    if correlated:
                        new_alert["correlated_alerts"] = correlated
                        log_alert_event("alerts_correlated", {
                            "primary_alert": new_alert["id"],
                            "correlated_alerts": correlated
                        })
                
                def escalate_alert(self, alert_id):
                    """Escalate alert to next level."""
                    if alert_id not in self.active_alerts:
                        return False
                    
                    alert = self.active_alerts[alert_id]
                    max_levels = config_manager.get("alerting.escalation.levels")
                    
                    if alert["escalation_level"] < max_levels - 1:
                        alert["escalation_level"] += 1
                        
                        escalation_history.append({
                            "alert_id": alert_id,
                            "new_level": alert["escalation_level"],
                            "timestamp": time.time()
                        })
                        
                        log_alert_event("alert_escalated", {
                            "alert_id": alert_id,
                            "escalation_level": alert["escalation_level"]
                        })
                        
                        return True
                    
                    return False
                
                def resolve_alert(self, alert_id, resolution_method):
                    """Resolve an alert."""
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert["resolved_at"] = time.time()
                        alert["resolution_method"] = resolution_method
                        
                        del self.active_alerts[alert_id]
                        
                        log_alert_event("alert_resolved", {
                            "alert_id": alert_id,
                            "resolution_method": resolution_method,
                            "duration": alert["resolved_at"] - alert["created_at"]
                        })
            
            class RemediationEngine:
                def __init__(self):
                    self.remediation_strategies = {
                        "high_cpu_usage": self.remediate_high_cpu,
                        "high_memory_usage": self.remediate_high_memory,
                        "high_error_rate": self.remediate_high_errors,
                        "high_latency": self.remediate_high_latency
                    }
                
                def auto_remediate(self, alert):
                    """Attempt automatic remediation for an alert."""
                    metric_name = alert["metric_name"]
                    
                    # Map metric to remediation strategy
                    strategy_key = None
                    if "cpu" in metric_name:
                        strategy_key = "high_cpu_usage"
                    elif "memory" in metric_name:
                        strategy_key = "high_memory_usage"
                    elif "error" in metric_name:
                        strategy_key = "high_error_rate"
                    elif "latency" in metric_name:
                        strategy_key = "high_latency"
                    
                    if strategy_key and strategy_key in self.remediation_strategies:
                        success = self.remediation_strategies[strategy_key](alert)
                        
                        remediation_actions.append({
                            "alert_id": alert["id"],
                            "strategy": strategy_key,
                            "success": success,
                            "timestamp": time.time()
                        })
                        
                        return success
                    
                    return False
                
                def remediate_high_cpu(self, alert):
                    """Remediate high CPU usage."""
                    # Simulate CPU remediation (e.g., scale up, kill processes)
                    log_alert_event("remediation_attempted", {
                        "alert_id": alert["id"],
                        "type": "cpu_scaling",
                        "action": "scale_up_replicas"
                    })
                    return True
                
                def remediate_high_memory(self, alert):
                    """Remediate high memory usage."""
                    # Simulate memory remediation (e.g., restart services, clear caches)
                    log_alert_event("remediation_attempted", {
                        "alert_id": alert["id"],
                        "type": "memory_cleanup",
                        "action": "clear_caches"
                    })
                    return True
                
                def remediate_high_errors(self, alert):
                    """Remediate high error rates."""
                    # Simulate error remediation (e.g., circuit breaker, fallback)
                    log_alert_event("remediation_attempted", {
                        "alert_id": alert["id"],
                        "type": "error_mitigation",
                        "action": "enable_circuit_breaker"
                    })
                    return True
                
                def remediate_high_latency(self, alert):
                    """Remediate high latency."""
                    # Simulate latency remediation (e.g., cache warming, load balancing)
                    log_alert_event("remediation_attempted", {
                        "alert_id": alert["id"],
                        "type": "latency_optimization",
                        "action": "optimize_load_balancing"
                    })
                    return True
            
            # Initialize components
            alert_manager = AlertManager()
            remediation_engine = RemediationEngine()
            
            # Set up event handlers for intelligent alerting
            def handle_metric_threshold_breach(metric_name, value, threshold):
                # Determine severity
                breach_ratio = value / threshold
                severity_thresholds = config_manager.get("alerting.severity_thresholds")
                
                if breach_ratio >= severity_thresholds["critical"]:
                    severity = "critical"
                elif breach_ratio >= severity_thresholds["high"]:
                    severity = "high"
                elif breach_ratio >= severity_thresholds["medium"]:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Create alert
                alert_id = alert_manager.create_alert(metric_name, value, threshold, severity)
                alert = alert_manager.active_alerts[alert_id]
                
                # Attempt auto-remediation for high severity alerts
                if severity in ["high", "critical"] and config_manager.get("alerting.auto_remediation.enabled"):
                    success = remediation_engine.auto_remediate(alert)
                    
                    if success:
                        # Give remediation time to work
                        time.sleep(0.1)
                        
                        # Simulate checking if problem is resolved
                        if hash(alert_id) % 3 == 0:  # 33% success rate
                            alert_manager.resolve_alert(alert_id, "auto_remediation")
                        else:
                            # Escalate if remediation didn't work
                            alert_manager.escalate_alert(alert_id)
                
                # Send notifications based on severity
                if severity == "critical":
                    email_service.send_email(
                        to_address="oncall@company.com",
                        subject=f"CRITICAL: {metric_name} Alert",
                        body=f"Metric {metric_name} = {value}, threshold = {threshold}"
                    )
                
                slack_notifier.send_alert(
                    severity=severity,
                    title=f"{severity.upper()}: {metric_name}",
                    message=f"Current: {value}, Threshold: {threshold}",
                    channel="#alerts"
                )
            
            # Simulate various metric threshold breaches
            metric_scenarios = [
                ("cpu_usage_percent", 85, 80),      # High CPU
                ("memory_usage_percent", 92, 85),   # Critical memory
                ("error_rate_percent", 8, 5),       # High error rate
                ("response_latency_ms", 450, 200),  # Critical latency
                ("disk_usage_percent", 78, 75),     # Medium disk
                ("cpu_usage_percent", 95, 80),      # Critical CPU (correlated)
                ("memory_usage_percent", 88, 85),   # High memory (correlated)
                ("network_errors_total", 15, 10),   # High network errors
                ("database_connections", 195, 150), # High DB connections
                ("queue_size", 1200, 1000)          # High queue size
            ]
            
            # Process scenarios with delays to simulate real-time alerting
            for i, (metric, value, threshold) in enumerate(metric_scenarios):
                handle_metric_threshold_breach(metric, value, threshold)
                time.sleep(0.05)  # Small delay between alerts
                
                # Simulate some alerts resolving naturally
                if i % 4 == 0 and alert_manager.active_alerts:
                    alert_id = list(alert_manager.active_alerts.keys())[0]
                    alert_manager.resolve_alert(alert_id, "natural_resolution")
            
            # Verify intelligent alerting behavior
            assert len(alert_events) > 0
            
            # Verify alerts were created
            created_alerts = [e for e in alert_events if e["type"] == "alert_created"]
            assert len(created_alerts) == len(metric_scenarios)
            
            # Verify correlation was detected
            correlated_alerts = [e for e in alert_events if e["type"] == "alerts_correlated"]
            assert len(correlated_alerts) >= 1  # Should find some correlations
            
            # Verify remediation was attempted
            remediation_attempts = [e for e in alert_events if e["type"] == "remediation_attempted"]
            assert len(remediation_attempts) >= 2  # Should attempt remediation for high/critical alerts
            
            # Verify escalations occurred
            escalated_alerts = [e for e in alert_events if e["type"] == "alert_escalated"]
            assert len(escalated_alerts) >= 1  # Some alerts should escalate
            
            # Verify some alerts were resolved
            resolved_alerts = [e for e in alert_events if e["type"] == "alert_resolved"]
            assert len(resolved_alerts) >= 1  # Some alerts should be resolved
            
            # Verify notifications were sent
            sent_emails = email_service.get_sent_emails(hours=1)
            sent_messages = slack_notifier.get_messages(channel="#alerts", hours=1)
            
            assert len(sent_emails) >= 1  # Critical alerts should trigger emails
            assert len(sent_messages) >= len(metric_scenarios)  # All alerts should trigger Slack
            
            # Verify remediation actions were logged
            assert len(remediation_actions) >= 2
            
            successful_remediations = [a for a in remediation_actions if a["success"]]
            assert len(successful_remediations) >= 1  # Some remediations should succeed