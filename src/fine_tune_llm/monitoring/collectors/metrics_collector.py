"""
Metrics collection module for training monitoring.

This module provides comprehensive metrics collection, storage,
and retrieval capabilities for training processes.
"""

import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging
from collections import defaultdict, deque

from ...core.interfaces import BaseComponent

logger = logging.getLogger(__name__)


class TrainingMetrics(BaseComponent):
    """Container for training metrics with thread-safe operations."""
    
    def __init__(self):
        """Initialize training metrics container."""
        self.metrics = defaultdict(deque)
        self.timestamps = deque()
        self.lock = threading.RLock()
        self.start_time = datetime.now()
        
        # Configuration
        self.max_history = 1000  # Maximum number of points to store
        self.update_interval = 5  # Seconds between updates
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.max_history = config.get('max_history', 1000)
        self.update_interval = config.get('update_interval', 5)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        with self.lock:
            self.metrics.clear()
            self.timestamps.clear()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "TrainingMetrics"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def add_metrics(self, metrics_dict: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Add metrics with thread safety.
        
        Args:
            metrics_dict: Dictionary of metric name -> value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            # Add timestamp
            self.timestamps.append(timestamp)
            
            # Add metrics
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.metrics[key].append(value)
                elif isinstance(value, dict):
                    # Handle nested metrics
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            self.metrics[f"{key}.{subkey}"].append(subvalue)
            
            # Limit history size
            while len(self.timestamps) > self.max_history:
                self.timestamps.popleft()
                for metric_deque in self.metrics.values():
                    if len(metric_deque) > self.max_history:
                        metric_deque.popleft()
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        with self.lock:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1]
            return None
    
    def get_history(self, metric_name: str, hours: Optional[int] = None) -> List[tuple]:
        """
        Get historical values for a metric.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours of history (None for all)
            
        Returns:
            List of (timestamp, value) tuples
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            timestamps = list(self.timestamps)
            values = list(self.metrics[metric_name])
            
            if hours is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                filtered_data = [
                    (ts, val) for ts, val in zip(timestamps, values)
                    if ts >= cutoff_time
                ]
                return filtered_data
            
            return list(zip(timestamps, values))
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame."""
        with self.lock:
            if not self.timestamps:
                return pd.DataFrame()
            
            data = {'timestamp': list(self.timestamps)}
            
            for metric_name, values in self.metrics.items():
                # Pad with None if lengths don't match
                metric_values = list(values)
                if len(metric_values) < len(self.timestamps):
                    metric_values.extend([None] * (len(self.timestamps) - len(metric_values)))
                elif len(metric_values) > len(self.timestamps):
                    metric_values = metric_values[-len(self.timestamps):]
                
                data[metric_name] = metric_values
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        with self.lock:
            stats = {}
            
            for metric_name, values in self.metrics.items():
                if values:
                    values_list = list(values)
                    stats[metric_name] = {
                        'count': len(values_list),
                        'mean': sum(values_list) / len(values_list),
                        'min': min(values_list),
                        'max': max(values_list),
                        'latest': values_list[-1] if values_list else 0.0
                    }
            
            return stats
    
    def save_to_file(self, filepath: Path):
        """Save metrics to JSON file."""
        with self.lock:
            data = {
                'start_time': self.start_time.isoformat(),
                'timestamps': [ts.isoformat() for ts in self.timestamps],
                'metrics': {
                    name: list(values) for name, values in self.metrics.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, filepath: Path):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        with self.lock:
            self.start_time = datetime.fromisoformat(data['start_time'])
            self.timestamps = deque([
                datetime.fromisoformat(ts) for ts in data['timestamps']
            ])
            
            self.metrics.clear()
            for name, values in data['metrics'].items():
                self.metrics[name] = deque(values)


class MetricsCollector(BaseComponent):
    """Advanced metrics collector with automatic monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.collector_config = self.config.get('collector', {})
        
        # Core components
        self.metrics = TrainingMetrics()
        
        # Monitoring settings
        self.monitoring = False
        self.monitor_thread = None
        self.update_callbacks = []
        
        # File monitoring
        self.monitor_files = []
        self.file_watchers = {}
        
        # Metrics sources
        self.metrics_sources = []  # List of callable functions
        
        # History settings
        self.auto_save = self.collector_config.get('auto_save', True)
        self.save_interval = self.collector_config.get('save_interval', 60)  # seconds
        self.save_path = Path(self.collector_config.get('save_path', 'artifacts/metrics/'))
        
        # Initialize metrics with config
        self.metrics.initialize(self.collector_config)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config.update(config)
        self.collector_config = self.config.get('collector', {})
        self.metrics.initialize(self.collector_config)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_monitoring()
        self.metrics.cleanup()
    
    @property
    def name(self) -> str:
        """Component name."""
        return "MetricsCollector"
    
    @property
    def version(self) -> str:
        """Component version."""
        return "2.0.0"
    
    def start_monitoring(self):
        """Start automatic metrics collection."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started metrics monitoring")
    
    def stop_monitoring(self):
        """Stop automatic metrics collection."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped metrics monitoring")
    
    def add_metrics_source(self, source_func: Callable[[], Dict[str, Any]]):
        """
        Add a metrics source function.
        
        Args:
            source_func: Function that returns metrics dictionary
        """
        self.metrics_sources.append(source_func)
    
    def add_file_monitor(self, filepath: Path, parser_func: Optional[Callable] = None):
        """
        Add file to monitor for metrics.
        
        Args:
            filepath: Path to file to monitor
            parser_func: Function to parse file content into metrics
        """
        self.monitor_files.append({
            'path': filepath,
            'parser': parser_func or self._default_json_parser,
            'last_modified': 0
        })
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add callback to be called when metrics are updated.
        
        Args:
            callback: Function to call with metrics dictionary
        """
        self.update_callbacks.append(callback)
    
    def collect_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Collect metrics from external source.
        
        Args:
            metrics_dict: Dictionary of metrics to collect
        """
        # Add to metrics container
        self.metrics.add_metrics(metrics_dict)
        
        # Call update callbacks
        for callback in self.update_callbacks:
            try:
                callback(metrics_dict)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
        
        # Auto-save if enabled
        if self.auto_save and hasattr(self, '_last_save_time'):
            time_since_save = time.time() - self._last_save_time
            if time_since_save >= self.save_interval:
                self._auto_save_metrics()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        self._last_save_time = time.time()
        
        while self.monitoring:
            try:
                self._update_metrics()
                time.sleep(self.metrics.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _update_metrics(self):
        """Update metrics from all sources."""
        all_metrics = {}
        
        # Collect from function sources
        for source_func in self.metrics_sources:
            try:
                metrics = source_func()
                if isinstance(metrics, dict):
                    all_metrics.update(metrics)
            except Exception as e:
                logger.error(f"Error collecting from metrics source: {e}")
        
        # Collect from file sources
        for file_info in self.monitor_files:
            try:
                filepath = file_info['path']
                if filepath.exists():
                    current_modified = filepath.stat().st_mtime
                    if current_modified > file_info['last_modified']:
                        metrics = file_info['parser'](filepath)
                        if isinstance(metrics, dict):
                            all_metrics.update(metrics)
                        file_info['last_modified'] = current_modified
            except Exception as e:
                logger.error(f"Error reading metrics file {file_info['path']}: {e}")
        
        # Collect metrics if any were found
        if all_metrics:
            self.collect_metrics(all_metrics)
    
    def _default_json_parser(self, filepath: Path) -> Dict[str, Any]:
        """Default JSON file parser."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _auto_save_metrics(self):
        """Auto-save metrics to file."""
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)
            save_file = self.save_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.metrics.save_to_file(save_file)
            self._last_save_time = time.time()
            logger.debug(f"Auto-saved metrics to {save_file}")
        except Exception as e:
            logger.error(f"Error auto-saving metrics: {e}")
    
    # Delegate methods to metrics container
    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """Add metrics directly."""
        self.collect_metrics(metrics_dict)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for metric."""
        return self.metrics.get_latest(metric_name)
    
    def get_history(self, metric_name: str, hours: Optional[int] = None) -> List[tuple]:
        """Get historical values for metric."""
        return self.metrics.get_history(metric_name, hours)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics as DataFrame."""
        return self.metrics.get_metrics_dataframe()
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics."""
        return self.metrics.get_summary_stats()