"""
Database adapter for hexagonal architecture.

This module provides database operations through the adapter pattern,
supporting multiple database backends with connection pooling and
transaction management.
"""

import sqlite3
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from contextlib import contextmanager
from datetime import datetime
import threading
import logging

from ..core.interfaces import DatabasePort
from ..core.exceptions import IntegrationError, DataError
from ..utils.resilience import circuit_breaker, retry

logger = logging.getLogger(__name__)


class DatabaseAdapter(DatabasePort):
    """
    Database adapter implementing database operations.
    
    Provides database access with connection pooling, transaction management,
    and resilience patterns for reliable data operations.
    """
    
    def __init__(self, 
                 database_url: str,
                 pool_size: int = 5,
                 enable_resilience: bool = True):
        """
        Initialize database adapter.
        
        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            enable_resilience: Enable circuit breaker and retry patterns
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.enable_resilience = enable_resilience
        
        # Connection pool (simple implementation for SQLite)
        self._connections = []
        self._connection_lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_pool()
        
        logger.info(f"Initialized DatabaseAdapter with URL: {database_url}")
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            with self._connection_lock:
                for _ in range(self.pool_size):
                    conn = sqlite3.connect(
                        self.database_url.replace('sqlite://', ''),
                        check_same_thread=False
                    )
                    conn.row_factory = sqlite3.Row  # Enable dict-like access
                    self._connections.append(conn)
            
            # Create basic schema
            self._create_schema()
            
        except Exception as e:
            raise IntegrationError(f"Failed to initialize database pool: {e}")
    
    def _create_schema(self):
        """Create basic database schema."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS error_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_id TEXT UNIQUE,
            timestamp REAL,
            exception_type TEXT,
            message TEXT,
            component TEXT,
            operation TEXT,
            severity TEXT,
            category TEXT,
            context TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT UNIQUE,
            model_name TEXT,
            model_type TEXT,
            version TEXT,
            config TEXT,
            metrics TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            model_id TEXT,
            config TEXT,
            status TEXT,
            metrics TEXT,
            start_time DATETIME,
            end_time DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_error_events_timestamp ON error_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_error_events_component ON error_events(component);
        CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name);
        CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
        """
        
        with self.get_connection() as conn:
            for statement in schema_sql.strip().split(';'):
                if statement.strip():
                    conn.execute(statement)
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool."""
        connection = None
        try:
            with self._connection_lock:
                if self._connections:
                    connection = self._connections.pop()
                else:
                    # Create new connection if pool is empty
                    connection = sqlite3.connect(
                        self.database_url.replace('sqlite://', ''),
                        check_same_thread=False
                    )
                    connection.row_factory = sqlite3.Row
            
            yield connection
            
        except Exception as e:
            if connection:
                try:
                    connection.rollback()
                except:
                    pass
            raise
        finally:
            if connection:
                try:
                    with self._connection_lock:
                        if len(self._connections) < self.pool_size:
                            self._connections.append(connection)
                        else:
                            connection.close()
                except:
                    pass
    
    @circuit_breaker("database_operations")
    @retry(max_attempts=3, base_delay=0.5)
    def execute_query(self, 
                     query: str, 
                     params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute SELECT query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params or ())
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                result = [dict(row) for row in rows]
                
            logger.debug(f"Executed query: {query[:50]}... (returned {len(result)} rows)")
            return result
            
        except Exception as e:
            raise IntegrationError(f"Database query failed: {e}")
    
    @circuit_breaker("database_operations")
    @retry(max_attempts=3, base_delay=0.5)
    def execute_command(self, 
                       command: str, 
                       params: Optional[Tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE command.
        
        Args:
            command: SQL command string
            params: Command parameters
            
        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(command, params or ())
                affected_rows = cursor.rowcount
                conn.commit()
                
            logger.debug(f"Executed command: {command[:50]}... ({affected_rows} rows affected)")
            return affected_rows
            
        except Exception as e:
            raise IntegrationError(f"Database command failed: {e}")
    
    def insert_error_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Insert error event into database.
        
        Args:
            event_data: Error event data
            
        Returns:
            True if successful
        """
        try:
            insert_sql = """
            INSERT OR REPLACE INTO error_events 
            (error_id, timestamp, exception_type, message, component, operation, 
             severity, category, context) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                event_data['error_id'],
                event_data['timestamp'],
                event_data['exception_type'],
                event_data['message'],
                event_data['component'],
                event_data['operation'],
                event_data['severity'],
                event_data['category'],
                json.dumps(event_data.get('context', {}))
            )
            
            affected = self.execute_command(insert_sql, params)
            return affected > 0
            
        except Exception as e:
            logger.error(f"Failed to insert error event: {e}")
            return False
    
    def get_error_events(self, 
                        component: Optional[str] = None,
                        severity: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve error events from database.
        
        Args:
            component: Filter by component
            severity: Filter by severity
            limit: Maximum number of events to return
            
        Returns:
            List of error events
        """
        try:
            conditions = []
            params = []
            
            if component:
                conditions.append("component = ?")
                params.append(component)
            
            if severity:
                conditions.append("severity = ?")
                params.append(severity)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT * FROM error_events 
            {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ?
            """
            params.append(limit)
            
            results = self.execute_query(query, tuple(params))
            
            # Parse context JSON
            for result in results:
                if result.get('context'):
                    try:
                        result['context'] = json.loads(result['context'])
                    except json.JSONDecodeError:
                        result['context'] = {}
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve error events: {e}")
            return []
    
    def insert_model_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Insert model metadata into database.
        
        Args:
            metadata: Model metadata
            
        Returns:
            True if successful
        """
        try:
            insert_sql = """
            INSERT OR REPLACE INTO model_metadata
            (model_id, model_name, model_type, version, config, metrics, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            params = (
                metadata['model_id'],
                metadata['model_name'],
                metadata['model_type'],
                metadata['version'],
                json.dumps(metadata.get('config', {})),
                json.dumps(metadata.get('metrics', {}))
            )
            
            affected = self.execute_command(insert_sql, params)
            return affected > 0
            
        except Exception as e:
            logger.error(f"Failed to insert model metadata: {e}")
            return False
    
    def get_model_metadata(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve model metadata from database.
        
        Args:
            model_id: Specific model ID to retrieve
            
        Returns:
            List of model metadata
        """
        try:
            if model_id:
                query = "SELECT * FROM model_metadata WHERE model_id = ?"
                params = (model_id,)
            else:
                query = "SELECT * FROM model_metadata ORDER BY updated_at DESC"
                params = ()
            
            results = self.execute_query(query, params)
            
            # Parse JSON fields
            for result in results:
                for field in ['config', 'metrics']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except json.JSONDecodeError:
                            result[field] = {}
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve model metadata: {e}")
            return []
    
    def insert_training_run(self, run_data: Dict[str, Any]) -> bool:
        """
        Insert training run data into database.
        
        Args:
            run_data: Training run data
            
        Returns:
            True if successful
        """
        try:
            insert_sql = """
            INSERT OR REPLACE INTO training_runs
            (run_id, model_id, config, status, metrics, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                run_data['run_id'],
                run_data['model_id'],
                json.dumps(run_data.get('config', {})),
                run_data['status'],
                json.dumps(run_data.get('metrics', {})),
                run_data.get('start_time'),
                run_data.get('end_time')
            )
            
            affected = self.execute_command(insert_sql, params)
            return affected > 0
            
        except Exception as e:
            logger.error(f"Failed to insert training run: {e}")
            return False
    
    def get_training_runs(self, 
                         model_id: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve training runs from database.
        
        Args:
            model_id: Filter by model ID
            status: Filter by status
            limit: Maximum number of runs to return
            
        Returns:
            List of training runs
        """
        try:
            conditions = []
            params = []
            
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT * FROM training_runs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """
            params.append(limit)
            
            results = self.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for result in results:
                for field in ['config', 'metrics']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except json.JSONDecodeError:
                            result[field] = {}
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve training runs: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {}
            
            # Count tables
            tables = ['error_events', 'model_metadata', 'training_runs']
            for table in tables:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = self.execute_query(count_query)
                stats[f"{table}_count"] = result[0]['count'] if result else 0
            
            # Recent activity
            recent_errors_query = """
            SELECT COUNT(*) as count FROM error_events 
            WHERE timestamp > ?
            """
            recent_time = (datetime.now().timestamp() - 3600)  # Last hour
            result = self.execute_query(recent_errors_query, (recent_time,))
            stats['recent_errors_count'] = result[0]['count'] if result else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def cleanup_old_records(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old records from database.
        
        Args:
            days_old: Delete records older than this many days
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_time = (datetime.now().timestamp() - (days_old * 24 * 3600))
            
            results = {}
            
            # Cleanup old error events
            delete_sql = "DELETE FROM error_events WHERE timestamp < ?"
            affected = self.execute_command(delete_sql, (cutoff_time,))
            results['error_events_deleted'] = affected
            
            # Cleanup old training runs (completed ones only)
            delete_training_sql = """
            DELETE FROM training_runs 
            WHERE created_at < datetime('now', '-' || ? || ' days')
            AND status IN ('completed', 'failed', 'cancelled')
            """
            affected = self.execute_command(delete_training_sql, (days_old,))
            results['training_runs_deleted'] = affected
            
            logger.info(f"Cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            return {}
    
    def close(self):
        """Close all database connections."""
        try:
            with self._connection_lock:
                while self._connections:
                    conn = self._connections.pop()
                    conn.close()
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")