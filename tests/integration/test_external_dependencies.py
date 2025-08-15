"""
Integration tests for external dependencies with comprehensive mocking.

This test module validates that all external dependencies are properly
mocked and that the system behaves correctly under various failure scenarios.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mocks import (
    MockHuggingFaceAPI, MockOpenAIAPI, MockDatabaseConnection,
    MockFileSystem, MockNetworkService, MockCacheService,
    MockTransformerModel, MockTokenizer, MockTrainingDataset,
    MockGPUManager, MockResourceMonitor, MockSecretManager,
    create_mock_environment, mock_dependencies_context,
    isolated_test_environment, mock_training_failure_scenario
)


class TestHuggingFaceIntegration:
    """Test HuggingFace API integration with mocking."""
    
    def test_model_loading_success(self):
        """Test successful model loading from HuggingFace."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            # Test model loading
            model = hf_api.from_pretrained('bert-base-uncased')
            
            assert model is not None
            assert model.model_name == 'bert-base-uncased'
            assert hasattr(model, 'config')
            assert len(hf_api.download_history) == 1
            assert hf_api.download_history[0]['type'] == 'model'
    
    def test_model_loading_failure(self):
        """Test model loading failure scenarios."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            hf_api.set_failure_rate(1.0)  # 100% failure rate
            
            with pytest.raises(ConnectionError):
                hf_api.from_pretrained('bert-base-uncased')
    
    def test_dataset_loading_success(self):
        """Test successful dataset loading."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            dataset = hf_api.load_dataset('imdb', size=1000)
            
            assert dataset is not None
            assert len(dataset) == 1000
            assert dataset.name == 'imdb'
            assert len(hf_api.download_history) == 1
            assert hf_api.download_history[0]['type'] == 'dataset'
    
    def test_dataset_loading_with_retry(self):
        """Test dataset loading with intermittent failures."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            hf_api.set_failure_rate(0.5)  # 50% failure rate
            
            # Try multiple times - some should succeed
            successes = 0
            failures = 0
            
            for _ in range(10):
                try:
                    dataset = hf_api.load_dataset('test-dataset')
                    successes += 1
                except ConnectionError:
                    failures += 1
            
            # Should have both successes and failures
            assert successes > 0
            assert failures > 0
    
    def test_model_push_to_hub(self):
        """Test pushing model to HuggingFace Hub."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            model = MockTransformerModel('test-model')
            
            result = hf_api.push_to_hub(model, 'test-user/test-model')
            
            assert 'repo_name' in result
            assert 'url' in result
            assert 'commit_sha' in result
            assert result['repo_name'] == 'test-user/test-model'
    
    def test_concurrent_model_loading(self):
        """Test concurrent model loading."""
        with mock_dependencies_context() as env:
            hf_api = env.get_service('huggingface')
            
            results = []
            errors = []
            
            def load_model(model_name):
                try:
                    model = hf_api.from_pretrained(model_name)
                    results.append(model)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple concurrent requests
            threads = []
            model_names = ['bert-base', 'gpt2', 'roberta-base', 'distilbert']
            
            for model_name in model_names:
                thread = threading.Thread(target=load_model, args=(model_name,))
                threads.append(thread)
                thread.start()
            
            # Wait for all to complete
            for thread in threads:
                thread.join()
            
            assert len(results) == 4
            assert len(errors) == 0
            assert len(hf_api.download_history) == 4


class TestOpenAIIntegration:
    """Test OpenAI API integration with mocking."""
    
    def test_chat_completion_success(self):
        """Test successful chat completion."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            messages = [
                {"role": "user", "content": "Hello, world!"}
            ]
            
            response = openai_api.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            assert 'id' in response
            assert 'choices' in response
            assert len(response['choices']) == 1
            assert 'message' in response['choices'][0]
            assert 'usage' in response
            assert len(openai_api.api_calls) == 1
    
    def test_chat_completion_failure(self):
        """Test chat completion failure scenarios."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            openai_api.set_failure_rate(1.0)
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(Exception):
                openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
    
    def test_embeddings_creation(self):
        """Test embeddings creation."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            texts = ["Hello world", "Test embedding"]
            
            response = openai_api.embeddings_create(
                model="text-embedding-ada-002",
                input_texts=texts
            )
            
            assert 'data' in response
            assert len(response['data']) == 2
            assert 'embedding' in response['data'][0]
            assert len(response['data'][0]['embedding']) == 1536  # OpenAI embedding size
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            openai_api.enable_rate_limiting(max_calls_per_minute=3)
            
            messages = [{"role": "user", "content": "Test"}]
            
            # First 3 calls should succeed
            for i in range(3):
                response = openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                assert response is not None
            
            # 4th call should fail due to rate limiting
            with pytest.raises(Exception, match="Rate limit exceeded"):
                openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
    
    def test_usage_tracking(self):
        """Test API usage tracking."""
        with mock_dependencies_context() as env:
            openai_api = env.get_service('openai')
            
            # Initial usage should be zero
            assert openai_api.usage_stats['requests'] == 0
            assert openai_api.usage_stats['tokens'] == 0
            
            # Make some API calls
            for i in range(3):
                openai_api.chat_completions_create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Test {i}"}]
                )
            
            # Usage should be tracked
            assert openai_api.usage_stats['requests'] == 3
            assert openai_api.usage_stats['tokens'] > 0
            assert openai_api.usage_stats['cost'] > 0


class TestDatabaseIntegration:
    """Test database integration with mocking."""
    
    def test_connection_lifecycle(self):
        """Test database connection lifecycle."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Test connection
            assert not db.is_connected
            success = db.connect()
            assert success
            assert db.is_connected
            
            # Test disconnection
            success = db.disconnect()
            assert success
            assert not db.is_connected
    
    def test_query_execution(self):
        """Test query execution."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Test SELECT query
            result = db.execute("SELECT * FROM users")
            assert isinstance(result, list)
            assert len(result) >= 0
            
            # Test INSERT query
            affected_rows = db.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                ("John Doe", "john@example.com")
            )
            assert affected_rows >= 0
            
            # Verify transaction was logged
            assert len(db.transactions) >= 2
            
            db.disconnect()
    
    def test_transaction_management(self):
        """Test database transaction management."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Test transaction
            db.begin_transaction()
            db.execute("INSERT INTO test (value) VALUES (?)", ("test1",))
            db.execute("INSERT INTO test (value) VALUES (?)", ("test2",))
            db.commit_transaction()
            
            # Verify transaction events were logged
            transaction_types = [t['type'] for t in db.transactions]
            assert 'BEGIN_TRANSACTION' in transaction_types
            assert 'COMMIT_TRANSACTION' in transaction_types
            
            db.disconnect()
    
    def test_batch_operations(self):
        """Test batch database operations."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Test batch insert
            params_list = [
                ("user1", "user1@example.com"),
                ("user2", "user2@example.com"),
                ("user3", "user3@example.com")
            ]
            
            results = db.execute_many(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                params_list
            )
            
            assert len(results) == 3
            assert len(db.transactions) == 3
            
            db.disconnect()
    
    def test_connection_failure_recovery(self):
        """Test database connection failure and recovery."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Test query without connection (should fail)
            with pytest.raises(Exception, match="Database not connected"):
                db.execute("SELECT * FROM test")
            
            # Connect and verify recovery
            db.connect()
            result = db.execute("SELECT * FROM test")
            assert result is not None
            
            db.disconnect()


class TestFileSystemIntegration:
    """Test file system integration with mocking."""
    
    def test_file_operations(self):
        """Test basic file operations."""
        with mock_dependencies_context() as env:
            fs = env.get_service('filesystem')
            
            # Test file writing and reading
            test_content = "This is test content"
            fs.write_file("/test/file.txt", test_content)
            
            assert fs.file_exists("/test/file.txt")
            content = fs.read_file("/test/file.txt")
            assert content == test_content
            
            # Test file size
            size = fs.get_file_size("/test/file.txt")
            assert size == len(test_content)
    
    def test_directory_operations(self):
        """Test directory operations."""
        with mock_dependencies_context() as env:
            fs = env.get_service('filesystem')
            
            # Create some files
            fs.write_file("/data/file1.txt", "content1")
            fs.write_file("/data/file2.txt", "content2")
            fs.write_file("/data/subdir/file3.txt", "content3")
            
            # Test directory listing
            files = fs.list_files("/data")
            assert len(files) >= 2
            assert any("file1.txt" in f for f in files)
            assert any("file2.txt" in f for f in files)
    
    def test_file_deletion(self):
        """Test file deletion."""
        with mock_dependencies_context() as env:
            fs = env.get_service('filesystem')
            
            # Create and delete file
            fs.write_file("/temp/deleteme.txt", "temporary content")
            assert fs.file_exists("/temp/deleteme.txt")
            
            fs.delete_file("/temp/deleteme.txt")
            assert not fs.file_exists("/temp/deleteme.txt")
    
    def test_file_not_found_errors(self):
        """Test file not found error handling."""
        with mock_dependencies_context() as env:
            fs = env.get_service('filesystem')
            
            # Test reading non-existent file
            with pytest.raises(FileNotFoundError):
                fs.read_file("/nonexistent/file.txt")
            
            # Test deleting non-existent file
            with pytest.raises(FileNotFoundError):
                fs.delete_file("/nonexistent/file.txt")
    
    def test_concurrent_file_operations(self):
        """Test concurrent file operations."""
        with mock_dependencies_context() as env:
            fs = env.get_service('filesystem')
            
            errors = []
            
            def write_files(worker_id):
                try:
                    for i in range(10):
                        fs.write_file(f"/worker_{worker_id}/file_{i}.txt", f"content_{worker_id}_{i}")
                except Exception as e:
                    errors.append(e)
            
            # Start multiple workers
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(target=write_files, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify no errors and files were created
            assert len(errors) == 0
            assert len(fs.operations) == 30  # 3 workers * 10 files


class TestNetworkIntegration:
    """Test network service integration with mocking."""
    
    def test_http_requests(self):
        """Test HTTP requests."""
        with mock_dependencies_context() as env:
            network = env.get_service('network')
            
            # Test GET request
            response = network.get("https://api.example.com/data")
            assert response.status_code == 200
            
            # Test POST request
            response = network.post(
                "https://api.example.com/submit",
                json={"key": "value"}
            )
            assert response.status_code == 200
            
            # Verify requests were logged
            assert len(network.requests) == 2
    
    def test_network_failures(self):
        """Test network failure scenarios."""
        with mock_dependencies_context() as env:
            network = env.get_service('network')
            network.set_failure_rate(1.0)
            
            with pytest.raises(ConnectionError):
                network.get("https://api.example.com/data")
    
    def test_custom_responses(self):
        """Test custom response configuration."""
        with mock_dependencies_context() as env:
            network = env.get_service('network')
            
            # Configure custom response
            network.add_response(
                "https://api.example.com/custom",
                {
                    "status_code": 201,
                    "json": {"message": "Created"},
                    "headers": {"Content-Type": "application/json"}
                }
            )
            
            response = network.get("https://api.example.com/custom")
            assert response.status_code == 201
            assert response.json()["message"] == "Created"
    
    def test_latency_simulation(self):
        """Test network latency simulation."""
        with mock_dependencies_context() as env:
            network = env.get_service('network')
            network.set_latency_range(0.5, 1.0)  # 500ms to 1s
            
            start_time = time.time()
            response = network.get("https://api.example.com/slow")
            end_time = time.time()
            
            # Should take at least 500ms
            assert (end_time - start_time) >= 0.5


class TestCacheIntegration:
    """Test cache service integration with mocking."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        with mock_dependencies_context() as env:
            cache = env.get_service('cache')
            
            # Test set and get
            cache.set("key1", "value1")
            value = cache.get("key1")
            assert value == "value1"
            
            # Test existence
            assert cache.exists("key1")
            assert not cache.exists("nonexistent")
            
            # Test deletion
            cache.delete("key1")
            assert not cache.exists("key1")
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        with mock_dependencies_context() as env:
            cache = env.get_service('cache')
            
            # Set with expiration
            cache.set("expiring_key", "value", expire_seconds=1)
            
            # Should exist immediately
            assert cache.exists("expiring_key")
            
            # Wait for expiration
            time.sleep(1.1)
            
            # Should be expired
            assert not cache.exists("expiring_key")
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        with mock_dependencies_context() as env:
            cache = env.get_service('cache')
            
            # Perform operations
            cache.set("key1", "value1")
            cache.get("key1")  # Hit
            cache.get("key2")  # Miss
            
            stats = cache.get_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["total_keys"] == 1
    
    def test_cache_clear(self):
        """Test cache clearing."""
        with mock_dependencies_context() as env:
            cache = env.get_service('cache')
            
            # Add some data
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            
            # Clear cache
            cache.clear()
            
            # Verify empty
            assert not cache.exists("key1")
            assert not cache.exists("key2")
            stats = cache.get_stats()
            assert stats["total_keys"] == 0


class TestGPUIntegration:
    """Test GPU management integration with mocking."""
    
    def test_gpu_discovery(self):
        """Test GPU discovery and information."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Test GPU count
            gpu_count = gpu_manager.get_gpu_count()
            assert gpu_count == 2
            
            # Test GPU info
            gpu_info = gpu_manager.get_gpu_info()
            assert len(gpu_info) == 2
            
            for gpu in gpu_info:
                assert 'id' in gpu
                assert 'name' in gpu
                assert 'memory_total' in gpu
                assert 'memory_used' in gpu
                assert 'memory_free' in gpu
    
    def test_memory_allocation(self):
        """Test GPU memory allocation."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Test memory allocation
            allocation_id = gpu_manager.allocate_memory(0, 1024, "test_process")
            assert allocation_id is not None
            
            # Verify allocation
            gpu_info = gpu_manager.get_gpu_info(0)
            assert len(gpu_info["processes"]) == 1
            assert gpu_info["processes"][0]["memory_mb"] == 1024
            
            # Test memory deallocation
            gpu_manager.free_memory(allocation_id)
            gpu_info = gpu_manager.get_gpu_info(0)
            assert len(gpu_info["processes"]) == 0
    
    def test_memory_usage_tracking(self):
        """Test GPU memory usage tracking."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Get initial usage
            initial_usage = gpu_manager.get_memory_usage(0)
            initial_used = initial_usage["used_mb"]
            
            # Allocate memory
            allocation_id = gpu_manager.allocate_memory(0, 2048, "test_process")
            
            # Check updated usage
            updated_usage = gpu_manager.get_memory_usage(0)
            assert updated_usage["used_mb"] == initial_used + 2048
            
            # Free memory
            gpu_manager.free_memory(allocation_id)
            
            # Check final usage
            final_usage = gpu_manager.get_memory_usage(0)
            assert final_usage["used_mb"] == initial_used
    
    def test_gpu_monitoring(self):
        """Test GPU monitoring."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Start monitoring
            gpu_manager.start_monitoring(interval_seconds=0.1)
            
            # Wait for some monitoring data
            time.sleep(0.3)
            
            # Stop monitoring
            gpu_manager.stop_monitoring()
            
            # Verify monitoring occurred (metrics should have changed)
            gpu_info = gpu_manager.get_gpu_info()
            assert len(gpu_info) == 2
    
    def test_gpu_failure_simulation(self):
        """Test GPU failure simulation."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Disable a GPU
            gpu_manager.set_gpu_availability(1, False)
            
            # Verify reduced GPU count
            available_count = gpu_manager.get_gpu_count()
            assert available_count == 1
            
            # Test allocation on disabled GPU should still work in mock
            # (in real scenario, this might fail)
            gpu_info = gpu_manager.get_gpu_info(1)
            assert not gpu_info["available"]


class TestResourceMonitoringIntegration:
    """Test resource monitoring integration."""
    
    def test_resource_metrics_collection(self):
        """Test resource metrics collection."""
        with mock_dependencies_context() as env:
            monitor = env.get_infrastructure('resource_monitor')
            
            # Get current metrics
            metrics = monitor.get_current_metrics()
            
            assert 'cpu' in metrics
            assert 'memory' in metrics
            assert 'disk' in metrics
            assert 'network' in metrics
            assert 'processes' in metrics
            
            # Verify metric structure
            assert 'percent' in metrics['cpu']
            assert 'total_gb' in metrics['memory']
            assert 'bytes_sent' in metrics['network']
    
    def test_monitoring_with_alerts(self):
        """Test monitoring with alert generation."""
        with mock_dependencies_context() as env:
            monitor = env.get_infrastructure('resource_monitor')
            
            # Set low thresholds to trigger alerts
            monitor.set_thresholds({
                'cpu_percent': 5,
                'memory_percent': 5,
                'disk_percent': 5
            })
            
            # Start monitoring
            monitor.start_monitoring(interval_seconds=0.1)
            
            # Wait for monitoring cycles
            time.sleep(0.3)
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Check for alerts
            alerts = monitor.get_alerts(hours=1)
            assert len(alerts) > 0  # Should have triggered some alerts
    
    def test_metrics_history(self):
        """Test metrics history collection."""
        with mock_dependencies_context() as env:
            monitor = env.get_infrastructure('resource_monitor')
            
            # Start monitoring
            monitor.start_monitoring(interval_seconds=0.1)
            
            # Wait for some data collection
            time.sleep(0.3)
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Check history
            history = monitor.get_metrics_history(hours=1)
            assert len(history) > 0
            
            # Verify history structure
            for entry in history[:3]:  # Check first 3 entries
                assert 'timestamp' in entry
                assert 'cpu' in entry
                assert 'memory' in entry


class TestIntegratedFailureScenarios:
    """Test integrated failure scenarios across multiple components."""
    
    def test_training_with_network_failures(self):
        """Test training pipeline with network failures."""
        with mock_training_failure_scenario() as env:
            # Get components
            hf_api = env.get_service('huggingface')
            openai_api = env.get_service('openai')
            
            # Test that some operations fail
            failure_count = 0
            success_count = 0
            
            for i in range(10):
                try:
                    model = hf_api.from_pretrained(f'test-model-{i}')
                    success_count += 1
                except ConnectionError:
                    failure_count += 1
            
            # Should have both successes and failures
            assert failure_count > 0
            assert success_count > 0
    
    def test_cascading_failures(self):
        """Test cascading failure scenarios."""
        with mock_dependencies_context() as env:
            # Simulate cascading failures
            env.simulate_failure('database', 1.0)
            env.simulate_failure('cache', 1.0)
            env.simulate_failure('network', 1.0)
            
            # Test that all services fail
            db = env.get_service('database')
            cache = env.get_service('cache')
            network = env.get_service('network')
            
            # Database operations should fail
            with pytest.raises(Exception):
                db.connect()
                db.execute("SELECT * FROM test")
            
            # Network operations should fail
            with pytest.raises(ConnectionError):
                network.get("https://example.com")
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        with mock_dependencies_context() as env:
            # Start with failures
            env.simulate_failure('huggingface', 1.0)
            hf_api = env.get_service('huggingface')
            
            # Verify failures
            with pytest.raises(ConnectionError):
                hf_api.from_pretrained('test-model')
            
            # "Fix" the service
            env.simulate_failure('huggingface', 0.0)
            
            # Should work now
            model = hf_api.from_pretrained('test-model')
            assert model is not None
    
    def test_resource_exhaustion_scenario(self):
        """Test resource exhaustion scenarios."""
        with mock_dependencies_context() as env:
            gpu_manager = env.get_infrastructure('gpu_manager')
            
            # Try to allocate more memory than available
            gpu_info = gpu_manager.get_gpu_info(0)
            total_memory = gpu_info['memory_total']
            
            # First allocation should succeed
            alloc1 = gpu_manager.allocate_memory(0, total_memory // 2, "process1")
            assert alloc1 is not None
            
            # Second large allocation should fail
            with pytest.raises(RuntimeError, match="Not enough memory"):
                gpu_manager.allocate_memory(0, total_memory, "process2")
            
            # Free memory and try again
            gpu_manager.free_memory(alloc1)
            alloc2 = gpu_manager.allocate_memory(0, total_memory // 2, "process2")
            assert alloc2 is not None
    
    def test_concurrent_operations_under_stress(self):
        """Test concurrent operations under stress conditions."""
        with mock_dependencies_context() as env:
            # Set moderate failure rates
            env.simulate_failure('huggingface', 0.3)
            env.simulate_failure('openai', 0.2)
            
            hf_api = env.get_service('huggingface')
            openai_api = env.get_service('openai')
            
            results = []
            errors = []
            
            def worker(worker_id):
                for i in range(5):
                    try:
                        # Try HuggingFace operation
                        model = hf_api.from_pretrained(f'model-{worker_id}-{i}')
                        results.append(('hf_success', worker_id, i))
                        
                        # Try OpenAI operation
                        response = openai_api.chat_completions_create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": f"Test {worker_id}-{i}"}]
                        )
                        results.append(('openai_success', worker_id, i))
                        
                    except Exception as e:
                        errors.append((worker_id, i, str(e)))
            
            # Start multiple workers
            threads = []
            for worker_id in range(4):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Should have both successes and errors
            assert len(results) > 0
            assert len(errors) > 0
            
            # Verify we attempted all operations
            total_attempts = len(results) + len(errors)
            assert total_attempts == 4 * 5 * 2  # 4 workers * 5 iterations * 2 operations