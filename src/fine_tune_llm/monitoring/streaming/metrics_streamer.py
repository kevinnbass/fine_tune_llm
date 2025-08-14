"""
Live Metrics Streaming System.

This module provides real-time streaming of metrics from training
to dashboards with WebSocket support and efficient buffering.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import threading
from queue import Queue, Empty
import time
from collections import deque
import websocket
import zmq

logger = logging.getLogger(__name__)


class StreamProtocol(Enum):
    """Streaming protocol types."""
    WEBSOCKET = "websocket"
    ZMQ = "zmq"
    HTTP_SSE = "http_sse"
    GRPC = "grpc"
    MEMORY = "memory"


class StreamEvent(Enum):
    """Types of streaming events."""
    METRIC_UPDATE = "metric_update"
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_COMPLETE = "batch_complete"
    CHECKPOINT_SAVED = "checkpoint_saved"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class MetricUpdate:
    """Metric update message."""
    metric_name: str
    value: float
    step: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'step': self.step,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class StreamMessage:
    """Stream message container."""
    event: StreamEvent
    data: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    sequence: int = 0
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            'event': self.event.value,
            'data': self.data if not hasattr(self.data, 'to_dict') else self.data.to_dict(),
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'sequence': self.sequence
        }, default=str)


class MetricsBuffer:
    """
    Efficient metrics buffering system.
    
    Provides buffering, aggregation, and sampling for high-frequency metrics.
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 aggregation_window: int = 100,
                 sampling_rate: float = 1.0):
        """
        Initialize metrics buffer.
        
        Args:
            max_size: Maximum buffer size
            aggregation_window: Window for aggregation
            sampling_rate: Sampling rate (0-1)
        """
        self.max_size = max_size
        self.aggregation_window = aggregation_window
        self.sampling_rate = sampling_rate
        
        # Circular buffer for metrics
        self.buffer = deque(maxlen=max_size)
        
        # Aggregation cache
        self.aggregation_cache: Dict[str, List[float]] = {}
        
        # Statistics
        self.total_received = 0
        self.total_sampled = 0
        self.total_aggregated = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add(self, update: MetricUpdate) -> bool:
        """
        Add metric update to buffer.
        
        Args:
            update: Metric update
            
        Returns:
            True if added (after sampling)
        """
        with self._lock:
            self.total_received += 1
            
            # Apply sampling
            import random
            if random.random() > self.sampling_rate:
                return False
            
            self.total_sampled += 1
            
            # Add to aggregation cache
            if update.metric_name not in self.aggregation_cache:
                self.aggregation_cache[update.metric_name] = []
            
            self.aggregation_cache[update.metric_name].append(update.value)
            
            # Check if aggregation needed
            if len(self.aggregation_cache[update.metric_name]) >= self.aggregation_window:
                aggregated = self._aggregate(update.metric_name)
                self.buffer.append(aggregated)
                self.total_aggregated += 1
                return True
            
            # Add to buffer
            self.buffer.append(update)
            return True
    
    def get_batch(self, max_items: int = 100) -> List[MetricUpdate]:
        """
        Get batch of metrics from buffer.
        
        Args:
            max_items: Maximum items to return
            
        Returns:
            List of metric updates
        """
        with self._lock:
            batch = []
            
            for _ in range(min(max_items, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            
            return batch
    
    def _aggregate(self, metric_name: str) -> MetricUpdate:
        """Aggregate metrics for given name."""
        values = self.aggregation_cache[metric_name]
        
        # Compute aggregated value (mean for now)
        import numpy as np
        aggregated_value = np.mean(values)
        
        # Create aggregated update
        update = MetricUpdate(
            metric_name=f"{metric_name}_aggregated",
            value=aggregated_value,
            step=len(values),
            metadata={
                'aggregation_type': 'mean',
                'window_size': len(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }
        )
        
        # Clear cache
        self.aggregation_cache[metric_name] = []
        
        return update
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                'buffer_size': len(self.buffer),
                'max_size': self.max_size,
                'total_received': self.total_received,
                'total_sampled': self.total_sampled,
                'total_aggregated': self.total_aggregated,
                'sampling_rate': self.sampling_rate,
                'aggregation_window': self.aggregation_window
            }


class StreamConnection:
    """Base class for stream connections."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize connection."""
        self.config = config
        self.connected = False
        self.error_count = 0
        self.last_error = None
    
    def connect(self) -> bool:
        """Establish connection."""
        raise NotImplementedError
    
    def disconnect(self):
        """Close connection."""
        raise NotImplementedError
    
    def send(self, message: StreamMessage) -> bool:
        """Send message."""
        raise NotImplementedError
    
    def receive(self) -> Optional[StreamMessage]:
        """Receive message."""
        raise NotImplementedError


class WebSocketConnection(StreamConnection):
    """WebSocket stream connection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize WebSocket connection."""
        super().__init__(config)
        self.ws = None
        self.url = config.get('url', 'ws://localhost:8765')
    
    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self.ws = websocket.create_connection(self.url)
            self.connected = True
            logger.info(f"Connected to WebSocket: {self.url}")
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            try:
                self.ws.close()
                self.connected = False
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")
    
    def send(self, message: StreamMessage) -> bool:
        """Send message via WebSocket."""
        if not self.connected or not self.ws:
            return False
        
        try:
            self.ws.send(message.to_json())
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"WebSocket send failed: {e}")
            return False
    
    def receive(self) -> Optional[StreamMessage]:
        """Receive message from WebSocket."""
        if not self.connected or not self.ws:
            return None
        
        try:
            data = self.ws.recv()
            # Parse and return as StreamMessage
            parsed = json.loads(data)
            return StreamMessage(
                event=StreamEvent(parsed.get('event', 'info')),
                data=parsed.get('data'),
                source=parsed.get('source', 'unknown'),
                timestamp=datetime.fromisoformat(parsed.get('timestamp', datetime.now().isoformat())),
                sequence=parsed.get('sequence', 0)
            )
            
        except Exception as e:
            logger.error(f"WebSocket receive failed: {e}")
            return None


class ZMQConnection(StreamConnection):
    """ZeroMQ stream connection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ZMQ connection."""
        super().__init__(config)
        self.context = None
        self.socket = None
        self.endpoint = config.get('endpoint', 'tcp://localhost:5555')
        self.socket_type = config.get('socket_type', 'PUB')
    
    def connect(self) -> bool:
        """Connect ZMQ socket."""
        try:
            self.context = zmq.Context()
            
            # Create appropriate socket type
            if self.socket_type == 'PUB':
                self.socket = self.context.socket(zmq.PUB)
                self.socket.bind(self.endpoint)
            elif self.socket_type == 'SUB':
                self.socket = self.context.socket(zmq.SUB)
                self.socket.connect(self.endpoint)
                self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
            elif self.socket_type == 'PUSH':
                self.socket = self.context.socket(zmq.PUSH)
                self.socket.bind(self.endpoint)
            elif self.socket_type == 'PULL':
                self.socket = self.context.socket(zmq.PULL)
                self.socket.connect(self.endpoint)
            
            self.connected = True
            logger.info(f"ZMQ {self.socket_type} connected: {self.endpoint}")
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"ZMQ connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect ZMQ socket."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        logger.info("ZMQ disconnected")
    
    def send(self, message: StreamMessage) -> bool:
        """Send message via ZMQ."""
        if not self.connected or not self.socket:
            return False
        
        try:
            self.socket.send_string(message.to_json())
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"ZMQ send failed: {e}")
            return False
    
    def receive(self) -> Optional[StreamMessage]:
        """Receive message from ZMQ."""
        if not self.connected or not self.socket:
            return None
        
        try:
            # Non-blocking receive
            data = self.socket.recv_string(flags=zmq.NOBLOCK)
            parsed = json.loads(data)
            return StreamMessage(
                event=StreamEvent(parsed.get('event', 'info')),
                data=parsed.get('data'),
                source=parsed.get('source', 'unknown'),
                timestamp=datetime.fromisoformat(parsed.get('timestamp', datetime.now().isoformat())),
                sequence=parsed.get('sequence', 0)
            )
            
        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.error(f"ZMQ receive failed: {e}")
            return None


class MemoryConnection(StreamConnection):
    """In-memory stream connection for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize memory connection."""
        super().__init__(config)
        self.queue = Queue(maxsize=config.get('max_size', 10000))
    
    def connect(self) -> bool:
        """Connect (always succeeds for memory)."""
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect."""
        self.connected = False
    
    def send(self, message: StreamMessage) -> bool:
        """Send message to queue."""
        if not self.connected:
            return False
        
        try:
            self.queue.put_nowait(message)
            return True
        except:
            return False
    
    def receive(self) -> Optional[StreamMessage]:
        """Receive message from queue."""
        if not self.connected:
            return None
        
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class MetricsStreamer:
    """
    Main metrics streaming coordinator.
    
    Manages connections, buffering, and streaming of metrics
    from training to dashboards.
    """
    
    def __init__(self, 
                 protocol: StreamProtocol = StreamProtocol.WEBSOCKET,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics streamer.
        
        Args:
            protocol: Streaming protocol
            config: Connection configuration
        """
        self.protocol = protocol
        self.config = config or {}
        
        # Connection
        self.connection: Optional[StreamConnection] = None
        
        # Metrics buffer
        self.buffer = MetricsBuffer(
            max_size=self.config.get('buffer_size', 10000),
            aggregation_window=self.config.get('aggregation_window', 100),
            sampling_rate=self.config.get('sampling_rate', 1.0)
        )
        
        # Subscribers
        self.subscribers: Set[Callable] = set()
        
        # Stream worker
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_running = False
        
        # Statistics
        self.total_sent = 0
        self.total_failed = 0
        self.sequence_counter = 0
        
        # Initialize connection
        self._initialize_connection()
        
        logger.info(f"Initialized MetricsStreamer with {protocol.value} protocol")
    
    def _initialize_connection(self):
        """Initialize stream connection based on protocol."""
        if self.protocol == StreamProtocol.WEBSOCKET:
            self.connection = WebSocketConnection(self.config)
        elif self.protocol == StreamProtocol.ZMQ:
            self.connection = ZMQConnection(self.config)
        elif self.protocol == StreamProtocol.MEMORY:
            self.connection = MemoryConnection(self.config)
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
    
    def start(self) -> bool:
        """
        Start metrics streaming.
        
        Returns:
            True if started successfully
        """
        try:
            # Connect
            if not self.connection.connect():
                return False
            
            # Start worker thread
            self.worker_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            # Send start event
            self._send_event(StreamEvent.TRAINING_START, {
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            })
            
            logger.info("Metrics streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop(self):
        """Stop metrics streaming."""
        try:
            # Send end event
            self._send_event(StreamEvent.TRAINING_END, {
                'timestamp': datetime.now().isoformat(),
                'total_sent': self.total_sent
            })
            
            # Stop worker
            self.worker_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            
            # Disconnect
            if self.connection:
                self.connection.disconnect()
            
            logger.info("Metrics streaming stopped")
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
    
    def send_metric(self, 
                   name: str,
                   value: float,
                   step: int,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Send metric update.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            metadata: Additional metadata
        """
        update = MetricUpdate(
            metric_name=name,
            value=value,
            step=step,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.buffer.add(update)
        
        # Notify subscribers
        self._notify_subscribers(update)
    
    def send_batch_metrics(self, metrics: Dict[str, float], step: int):
        """
        Send batch of metrics.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Training step
        """
        for name, value in metrics.items():
            self.send_metric(name, value, step)
    
    def subscribe(self, callback: Callable) -> bool:
        """
        Subscribe to metric updates.
        
        Args:
            callback: Callback function
            
        Returns:
            True if subscribed successfully
        """
        self.subscribers.add(callback)
        return True
    
    def unsubscribe(self, callback: Callable) -> bool:
        """
        Unsubscribe from metric updates.
        
        Args:
            callback: Callback function
            
        Returns:
            True if unsubscribed successfully
        """
        self.subscribers.discard(callback)
        return True
    
    def _worker_loop(self):
        """Worker loop for streaming metrics."""
        while self.worker_running:
            try:
                # Get batch from buffer
                batch = self.buffer.get_batch(max_items=50)
                
                if batch:
                    # Send batch
                    for update in batch:
                        success = self._send_metric_update(update)
                        if success:
                            self.total_sent += 1
                        else:
                            self.total_failed += 1
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _send_metric_update(self, update: MetricUpdate) -> bool:
        """Send single metric update."""
        message = StreamMessage(
            event=StreamEvent.METRIC_UPDATE,
            data=update,
            source="training",
            sequence=self._get_next_sequence()
        )
        
        return self.connection.send(message)
    
    def _send_event(self, event: StreamEvent, data: Any):
        """Send stream event."""
        message = StreamMessage(
            event=event,
            data=data,
            source="training",
            sequence=self._get_next_sequence()
        )
        
        self.connection.send(message)
    
    def _notify_subscribers(self, update: MetricUpdate):
        """Notify all subscribers of metric update."""
        for callback in self.subscribers:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        stats = {
            'protocol': self.protocol.value,
            'connected': self.connection.connected if self.connection else False,
            'total_sent': self.total_sent,
            'total_failed': self.total_failed,
            'sequence': self.sequence_counter,
            'buffer_stats': self.buffer.get_statistics()
        }
        
        if self.connection:
            stats['connection_errors'] = self.connection.error_count
            stats['last_error'] = self.connection.last_error
        
        return stats


# Global streamer instance
_metrics_streamer = None

def get_metrics_streamer(protocol: StreamProtocol = StreamProtocol.WEBSOCKET,
                        config: Optional[Dict[str, Any]] = None) -> MetricsStreamer:
    """Get global metrics streamer instance."""
    global _metrics_streamer
    if _metrics_streamer is None:
        _metrics_streamer = MetricsStreamer(protocol, config)
    return _metrics_streamer


# Convenience functions

def start_metrics_streaming(protocol: StreamProtocol = StreamProtocol.WEBSOCKET,
                          config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Start metrics streaming.
    
    Args:
        protocol: Streaming protocol
        config: Connection configuration
        
    Returns:
        True if started successfully
    """
    streamer = get_metrics_streamer(protocol, config)
    return streamer.start()


def stop_metrics_streaming():
    """Stop metrics streaming."""
    global _metrics_streamer
    if _metrics_streamer:
        _metrics_streamer.stop()
        _metrics_streamer = None


def stream_metric(name: str, value: float, step: int, metadata: Optional[Dict[str, Any]] = None):
    """
    Stream single metric.
    
    Args:
        name: Metric name
        value: Metric value
        step: Training step
        metadata: Additional metadata
    """
    streamer = get_metrics_streamer()
    streamer.send_metric(name, value, step, metadata)