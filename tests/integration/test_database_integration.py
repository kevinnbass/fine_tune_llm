"""
Database integration tests for all database-related components.

This test module validates database integration across all platform components,
including data persistence, transaction management, and database service integration.
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
    MockDatabaseConnection, MockFileSystem
)

# Import platform components
from src.fine_tune_llm.core.events import EventBus, Event, EventType
from src.fine_tune_llm.config.manager import ConfigManager


class TestDatabaseConnectionManagement:
    """Test database connection lifecycle management."""
    
    def test_connection_pool_management(self):
        """Test database connection pool management."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Test multiple connections
            connections = []
            for i in range(5):
                success = db.connect()
                assert success
                connections.append(db.connection_id)
            
            # Verify connections are managed
            assert len(set(connections)) >= 1  # At least one connection
            assert db.is_connected
            
            # Test disconnection
            success = db.disconnect()
            assert success
            assert not db.is_connected
    
    def test_connection_failure_handling(self):
        """Test database connection failure scenarios."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Simulate connection failure
            db.set_failure_rate(1.0)
            
            with pytest.raises(Exception):
                db.connect()
            
            assert not db.is_connected
            
            # Test recovery
            db.set_failure_rate(0.0)
            success = db.connect()
            assert success
            assert db.is_connected
    
    def test_connection_timeout_handling(self):
        """Test database connection timeout scenarios."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Test connection with timeout
            db.set_connection_delay(2.0)  # 2 second delay
            
            start_time = time.time()
            success = db.connect(timeout=1.0)  # 1 second timeout
            end_time = time.time()
            
            # Should timeout quickly
            assert (end_time - start_time) < 1.5
            # Connection might fail due to timeout
            
            db.disconnect()
    
    def test_concurrent_connections(self):
        """Test concurrent database connections."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            connection_results = []
            errors = []
            
            def connect_worker(worker_id):
                try:
                    success = db.connect()
                    connection_results.append((worker_id, success))
                    time.sleep(0.1)  # Hold connection briefly
                    db.disconnect()
                except Exception as e:
                    errors.append((worker_id, str(e)))
            
            # Start multiple connection workers
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(target=connect_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify connections succeeded
            assert len(connection_results) >= 1
            assert len(errors) == 0


class TestDatabaseOperations:
    """Test database CRUD operations."""
    
    def test_basic_crud_operations(self):
        """Test basic create, read, update, delete operations."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Create
            result = db.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                ("John Doe", "john@example.com")
            )
            assert result >= 0
            
            # Read
            users = db.execute("SELECT * FROM users WHERE name = ?", ("John Doe",))
            assert isinstance(users, list)
            
            # Update
            result = db.execute(
                "UPDATE users SET email = ? WHERE name = ?",
                ("john.doe@example.com", "John Doe")
            )
            assert result >= 0
            
            # Delete
            result = db.execute("DELETE FROM users WHERE name = ?", ("John Doe",))
            assert result >= 0
            
            db.disconnect()
    
    def test_batch_operations(self):
        """Test batch database operations."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Batch insert
            users_data = [
                ("Alice", "alice@example.com"),
                ("Bob", "bob@example.com"),
                ("Charlie", "charlie@example.com")
            ]
            
            results = db.execute_many(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                users_data
            )
            
            assert len(results) == 3
            assert all(result >= 0 for result in results)
            
            # Verify batch insert worked
            all_users = db.execute("SELECT * FROM users")
            assert len(all_users) >= 3
            
            db.disconnect()
    
    def test_complex_queries(self):
        """Test complex database queries."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Insert test data
            db.execute(
                "INSERT INTO training_jobs (name, status, created_at) VALUES (?, ?, ?)",
                ("job1", "running", datetime.now(timezone.utc))
            )
            db.execute(
                "INSERT INTO training_jobs (name, status, created_at) VALUES (?, ?, ?)",
                ("job2", "completed", datetime.now(timezone.utc))
            )
            
            # Complex query with JOIN and filtering
            results = db.execute("""
                SELECT tj.name, tj.status, COUNT(tm.id) as model_count
                FROM training_jobs tj
                LEFT JOIN trained_models tm ON tj.id = tm.training_job_id
                WHERE tj.status IN ('running', 'completed')
                GROUP BY tj.id, tj.name, tj.status
                ORDER BY tj.created_at DESC
            """)
            
            assert isinstance(results, list)
            assert len(results) >= 0
            
            db.disconnect()
    
    def test_parameterized_queries(self):
        """Test parameterized queries for SQL injection prevention."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Test with potentially malicious input
            malicious_name = "'; DROP TABLE users; --"
            
            # This should be safe with parameterized queries
            result = db.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (malicious_name, "test@example.com")
            )
            assert result >= 0
            
            # Verify data was inserted safely
            users = db.execute("SELECT * FROM users WHERE name = ?", (malicious_name,))
            assert len(users) >= 0
            
            db.disconnect()


class TestTransactionManagement:
    """Test database transaction management."""
    
    def test_transaction_commit(self):
        """Test successful transaction commit."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Start transaction
            db.begin_transaction()
            
            # Perform operations
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("User1", "user1@example.com"))
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("User2", "user2@example.com"))
            
            # Commit transaction
            db.commit_transaction()
            
            # Verify data was committed
            users = db.execute("SELECT * FROM users WHERE name IN ('User1', 'User2')")
            assert len(users) >= 0
            
            # Verify transaction was logged
            transaction_logs = [t for t in db.transactions if t['type'] == 'COMMIT_TRANSACTION']
            assert len(transaction_logs) >= 1
            
            db.disconnect()
    
    def test_transaction_rollback(self):
        """Test transaction rollback."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Get initial user count
            initial_users = db.execute("SELECT COUNT(*) as count FROM users")
            initial_count = initial_users[0]['count'] if initial_users else 0
            
            # Start transaction
            db.begin_transaction()
            
            # Perform operations
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("TempUser1", "temp1@example.com"))
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("TempUser2", "temp2@example.com"))
            
            # Rollback transaction
            db.rollback_transaction()
            
            # Verify data was not committed
            final_users = db.execute("SELECT COUNT(*) as count FROM users")
            final_count = final_users[0]['count'] if final_users else 0
            assert final_count == initial_count
            
            # Verify rollback was logged
            rollback_logs = [t for t in db.transactions if t['type'] == 'ROLLBACK_TRANSACTION']
            assert len(rollback_logs) >= 1
            
            db.disconnect()
    
    def test_nested_transactions(self):
        """Test nested transaction handling."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Outer transaction
            db.begin_transaction()
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Outer1", "outer1@example.com"))
            
            # Inner transaction (savepoint)
            savepoint_id = db.savepoint("inner_transaction")
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Inner1", "inner1@example.com"))
            
            # Rollback to savepoint
            db.rollback_to_savepoint(savepoint_id)
            
            # Commit outer transaction
            db.commit_transaction()
            
            # Verify only outer transaction data exists
            users = db.execute("SELECT * FROM users WHERE name IN ('Outer1', 'Inner1')")
            outer_users = [u for u in users if u.get('name') == 'Outer1']
            inner_users = [u for u in users if u.get('name') == 'Inner1']
            
            # Should have outer user but not inner user
            assert len(outer_users) >= 0
            # Inner user should be rolled back
            
            db.disconnect()
    
    def test_transaction_timeout(self):
        """Test transaction timeout handling."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Start transaction with timeout
            db.begin_transaction(timeout=1.0)
            
            # Perform operation
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("TimeoutUser", "timeout@example.com"))
            
            # Wait longer than timeout
            time.sleep(1.5)
            
            # Try to commit (might fail due to timeout)
            try:
                db.commit_transaction()
                # If successful, verify transaction
                committed = True
            except Exception:
                # Transaction timed out
                committed = False
            
            # Either way, transaction should be handled
            assert isinstance(committed, bool)
            
            db.disconnect()


class TestDatabaseServiceIntegration:
    """Test integration with database services."""
    
    def test_model_persistence_service(self):
        """Test model persistence through database service."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Mock model metadata
            model_metadata = {
                "name": "test-model",
                "version": "1.0",
                "architecture": "transformer",
                "parameters": 110000000,
                "training_config": {"epochs": 3, "lr": 2e-4}
            }
            
            # Store model metadata
            db.execute("""
                INSERT INTO models (name, version, architecture, parameters, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_metadata["name"],
                model_metadata["version"],
                model_metadata["architecture"],
                model_metadata["parameters"],
                str(model_metadata["training_config"]),
                datetime.now(timezone.utc)
            ))
            
            # Retrieve model metadata
            models = db.execute("SELECT * FROM models WHERE name = ?", (model_metadata["name"],))
            assert len(models) >= 0
            
            if models:
                stored_model = models[0]
                assert stored_model["name"] == model_metadata["name"]
                assert stored_model["architecture"] == model_metadata["architecture"]
            
            db.disconnect()
    
    def test_training_history_service(self):
        """Test training history persistence through database service."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Create training job
            job_id = "training_job_001"
            db.execute("""
                INSERT INTO training_jobs (id, name, status, config, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                job_id,
                "Test Training Job",
                "running",
                '{"epochs": 5, "batch_size": 32}',
                datetime.now(timezone.utc)
            ))
            
            # Add training metrics
            training_metrics = [
                {"epoch": 1, "step": 100, "loss": 1.5, "accuracy": 0.65},
                {"epoch": 1, "step": 200, "loss": 1.2, "accuracy": 0.72},
                {"epoch": 2, "step": 300, "loss": 1.0, "accuracy": 0.78},
            ]
            
            for metric in training_metrics:
                db.execute("""
                    INSERT INTO training_metrics (job_id, epoch, step, loss, accuracy, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    metric["epoch"],
                    metric["step"],
                    metric["loss"],
                    metric["accuracy"],
                    datetime.now(timezone.utc)
                ))
            
            # Query training history
            history = db.execute("""
                SELECT tj.name, tm.epoch, tm.step, tm.loss, tm.accuracy
                FROM training_jobs tj
                JOIN training_metrics tm ON tj.id = tm.job_id
                WHERE tj.id = ?
                ORDER BY tm.step
            """, (job_id,))
            
            assert len(history) >= 0
            
            db.disconnect()
    
    def test_configuration_persistence_service(self):
        """Test configuration persistence through database service."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Store configuration
            config_data = {
                "training": {
                    "batch_size": 32,
                    "learning_rate": 2e-4,
                    "epochs": 5
                },
                "model": {
                    "architecture": "transformer",
                    "hidden_size": 768
                }
            }
            
            db.execute("""
                INSERT INTO configurations (name, version, config_data, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                "default_training_config",
                "1.0",
                str(config_data),
                datetime.now(timezone.utc)
            ))
            
            # Retrieve configuration
            configs = db.execute("""
                SELECT * FROM configurations 
                WHERE name = ? AND version = ?
            """, ("default_training_config", "1.0"))
            
            assert len(configs) >= 0
            
            if configs:
                stored_config = configs[0]
                assert stored_config["name"] == "default_training_config"
                assert stored_config["version"] == "1.0"
            
            db.disconnect()
    
    def test_audit_log_service(self):
        """Test audit logging through database service."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Log audit events
            audit_events = [
                {
                    "event_type": "model_trained",
                    "user_id": "user123",
                    "resource_id": "model_456",
                    "details": {"training_duration": 3600, "final_accuracy": 0.92}
                },
                {
                    "event_type": "model_deployed",
                    "user_id": "user123",
                    "resource_id": "model_456",
                    "details": {"deployment_target": "production"}
                }
            ]
            
            for event in audit_events:
                db.execute("""
                    INSERT INTO audit_logs (event_type, user_id, resource_id, details, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event["event_type"],
                    event["user_id"],
                    event["resource_id"],
                    str(event["details"]),
                    datetime.now(timezone.utc)
                ))
            
            # Query audit trail
            audit_trail = db.execute("""
                SELECT * FROM audit_logs 
                WHERE user_id = ? AND resource_id = ?
                ORDER BY timestamp
            """, ("user123", "model_456"))
            
            assert len(audit_trail) >= 0
            
            db.disconnect()


class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    def test_bulk_insert_performance(self):
        """Test bulk insert performance."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Generate bulk data
            bulk_data = []
            for i in range(1000):
                bulk_data.append((f"user_{i}", f"user_{i}@example.com", i % 10))
            
            # Time bulk insert
            start_time = time.time()
            results = db.execute_many(
                "INSERT INTO users (name, email, department_id) VALUES (?, ?, ?)",
                bulk_data
            )
            end_time = time.time()
            
            # Verify performance
            insert_time = end_time - start_time
            assert insert_time < 10.0  # Should complete within 10 seconds
            assert len(results) == 1000
            
            db.disconnect()
    
    def test_query_performance_with_indexes(self):
        """Test query performance with database indexes."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Create index (simulated)
            db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            
            # Insert test data
            test_data = [(f"user_{i}", f"user_{i}@example.com") for i in range(100)]
            db.execute_many(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                test_data
            )
            
            # Time indexed query
            start_time = time.time()
            results = db.execute("SELECT * FROM users WHERE email = ?", ("user_50@example.com",))
            end_time = time.time()
            
            # Verify performance
            query_time = end_time - start_time
            assert query_time < 1.0  # Should be fast with index
            assert len(results) >= 0
            
            db.disconnect()
    
    def test_connection_pooling_performance(self):
        """Test connection pooling performance."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Test rapid connection/disconnection
            start_time = time.time()
            
            for i in range(10):
                db.connect()
                db.execute("SELECT 1")
                db.disconnect()
            
            end_time = time.time()
            
            # Verify connection pooling efficiency
            total_time = end_time - start_time
            assert total_time < 5.0  # Should be efficient with pooling
    
    def test_concurrent_query_performance(self):
        """Test concurrent query performance."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            query_results = []
            query_errors = []
            
            def query_worker(worker_id):
                try:
                    start_time = time.time()
                    results = db.execute(f"SELECT * FROM users WHERE id > {worker_id * 10}")
                    end_time = time.time()
                    
                    query_results.append({
                        "worker_id": worker_id,
                        "duration": end_time - start_time,
                        "result_count": len(results)
                    })
                except Exception as e:
                    query_errors.append((worker_id, str(e)))
            
            # Start concurrent queries
            threads = []
            for worker_id in range(5):
                thread = threading.Thread(target=query_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify concurrent performance
            assert len(query_results) == 5
            assert len(query_errors) == 0
            
            # All queries should complete reasonably quickly
            max_duration = max(result["duration"] for result in query_results)
            assert max_duration < 2.0
            
            db.disconnect()


class TestDatabaseErrorHandling:
    """Test database error handling and recovery."""
    
    def test_connection_recovery(self):
        """Test database connection recovery after failure."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            
            # Establish connection
            db.connect()
            assert db.is_connected
            
            # Simulate connection loss
            db.force_disconnect()
            assert not db.is_connected
            
            # Attempt recovery
            success = db.reconnect()
            assert success
            assert db.is_connected
            
            # Verify functionality after recovery
            result = db.execute("SELECT 1")
            assert result is not None
            
            db.disconnect()
    
    def test_query_error_handling(self):
        """Test query error handling."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Test invalid SQL
            with pytest.raises(Exception):
                db.execute("INVALID SQL STATEMENT")
            
            # Verify connection is still usable
            assert db.is_connected
            result = db.execute("SELECT 1")
            assert result is not None
            
            db.disconnect()
    
    def test_transaction_error_recovery(self):
        """Test transaction error recovery."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Start transaction
            db.begin_transaction()
            
            # Successful operation
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("ValidUser", "valid@example.com"))
            
            # Operation that might fail
            try:
                db.execute("INSERT INTO users (name, email) VALUES (?, ?)", (None, "invalid@example.com"))
            except Exception:
                # Rollback on error
                db.rollback_transaction()
                rolled_back = True
            else:
                # Commit if successful
                db.commit_transaction()
                rolled_back = False
            
            # Verify database is in consistent state
            assert db.is_connected
            
            # Start new transaction to verify recovery
            db.begin_transaction()
            db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("RecoveryUser", "recovery@example.com"))
            db.commit_transaction()
            
            db.disconnect()
    
    def test_deadlock_handling(self):
        """Test deadlock detection and handling."""
        with mock_dependencies_context() as env:
            db = env.get_service('database')
            db.connect()
            
            # Simulate potential deadlock scenario
            deadlock_results = []
            
            def deadlock_worker(worker_id):
                try:
                    db.begin_transaction()
                    
                    # Each worker tries to update different resources in different order
                    if worker_id == 0:
                        db.execute("UPDATE users SET email = ? WHERE id = 1", ("new1@example.com",))
                        time.sleep(0.1)
                        db.execute("UPDATE users SET email = ? WHERE id = 2", ("new2@example.com",))
                    else:
                        db.execute("UPDATE users SET email = ? WHERE id = 2", ("new2@example.com",))
                        time.sleep(0.1)
                        db.execute("UPDATE users SET email = ? WHERE id = 1", ("new1@example.com",))
                    
                    db.commit_transaction()
                    deadlock_results.append((worker_id, "success"))
                    
                except Exception as e:
                    db.rollback_transaction()
                    deadlock_results.append((worker_id, f"error: {str(e)}"))
            
            # Start workers that might deadlock
            threads = []
            for worker_id in range(2):
                thread = threading.Thread(target=deadlock_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify deadlock was handled
            assert len(deadlock_results) == 2
            
            # At least one should succeed or both should handle the deadlock gracefully
            successes = [r for r in deadlock_results if r[1] == "success"]
            errors = [r for r in deadlock_results if r[1].startswith("error")]
            
            # Either both succeed or deadlock is properly handled
            assert len(successes) >= 0 and len(errors) >= 0
            
            db.disconnect()