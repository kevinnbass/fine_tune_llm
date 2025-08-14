"""
Data Flow Coordination System.

This module provides centralized coordination of data flow between all
components with consistent interfaces, routing, and transformation.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import asyncio
from collections import deque
import json
from pathlib import Path

from ...core.events import EventBus, Event, EventType
from ..processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class DataFlowDirection(Enum):
    """Data flow directions."""
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"


class DataType(Enum):
    """Data types in the system."""
    RAW_TEXT = "raw_text"
    PROCESSED_TEXT = "processed_text"
    TRAINING_DATA = "training_data"
    MODEL_PREDICTIONS = "model_predictions"
    EVALUATION_RESULTS = "evaluation_results"
    METRICS = "metrics"
    CONFIGURATION = "configuration"
    EVENTS = "events"
    LOGS = "logs"
    CHECKPOINTS = "checkpoints"
    CUSTOM = "custom"


class FlowState(Enum):
    """Data flow states."""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class DataPacket:
    """Data packet for flow coordination."""
    data: Any
    data_type: DataType
    source: str
    destination: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    packet_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'packet_id': self.packet_id,
            'data_type': self.data_type.value,
            'source': self.source,
            'destination': self.destination,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'data_size': len(str(self.data)) if self.data else 0
        }


@dataclass
class FlowRoute:
    """Data flow route definition."""
    name: str
    source: str
    destination: str
    data_type: DataType
    processor: Optional[BaseProcessor] = None
    filter_fn: Optional[Callable[[DataPacket], bool]] = None
    transform_fn: Optional[Callable[[DataPacket], DataPacket]] = None
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowMetrics:
    """Metrics for data flow monitoring."""
    packets_processed: int = 0
    packets_failed: int = 0
    total_bytes: int = 0
    average_latency: float = 0.0
    current_throughput: float = 0.0
    error_rate: float = 0.0
    last_activity: Optional[datetime] = None


class DataBuffer:
    """Thread-safe data buffer for flow coordination."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize data buffer."""
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        
        # Statistics
        self.total_added = 0
        self.total_removed = 0
        self.total_dropped = 0
    
    def put(self, packet: DataPacket) -> bool:
        """Add packet to buffer."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                # Drop oldest packet
                dropped = self.buffer.popleft()
                self.total_dropped += 1
                logger.warning(f"Dropped packet {dropped.packet_id} - buffer full")
            
            self.buffer.append(packet)
            self.total_added += 1
            self.not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """Get packet from buffer."""
        with self.not_empty:
            if not self.buffer:
                if timeout is None:
                    self.not_empty.wait()
                else:
                    if not self.not_empty.wait(timeout):
                        return None
            
            if self.buffer:
                packet = self.buffer.popleft()
                self.total_removed += 1
                return packet
            
            return None
    
    def peek(self) -> Optional[DataPacket]:
        """Peek at next packet without removing."""
        with self.lock:
            return self.buffer[0] if self.buffer else None
    
    def size(self) -> int:
        """Get buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'size': len(self.buffer),
                'max_size': self.max_size,
                'total_added': self.total_added,
                'total_removed': self.total_removed,
                'total_dropped': self.total_dropped,
                'utilization': len(self.buffer) / self.max_size
            }


class DataFlowCoordinator:
    """
    Central data flow coordination system.
    
    Manages data flow between all components with routing,
    transformation, buffering, and monitoring capabilities.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize data flow coordinator.
        
        Args:
            event_bus: Event bus for notifications
        """
        self.event_bus = event_bus or EventBus()
        
        # Flow routes
        self.routes: Dict[str, FlowRoute] = {}
        
        # Component registry
        self.components: Dict[str, Any] = {}
        
        # Data buffers
        self.buffers: Dict[str, DataBuffer] = {}
        
        # Flow state
        self.state = FlowState.ACTIVE
        
        # Metrics
        self.metrics: Dict[str, FlowMetrics] = {}
        
        # Worker threads
        self.workers: Dict[str, threading.Thread] = {}
        self.worker_running = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default buffers
        self._initialize_default_buffers()
        
        # Initialize default routes
        self._initialize_default_routes()
        
        logger.info("Initialized DataFlowCoordinator")
    
    def _initialize_default_buffers(self):
        """Initialize default data buffers."""
        buffer_configs = [
            ('training_data', 5000),
            ('inference_data', 2000),
            ('evaluation_data', 1000),
            ('metrics', 10000),
            ('logs', 20000),
            ('events', 5000),
            ('checkpoints', 100)
        ]
        
        for buffer_name, max_size in buffer_configs:
            self.buffers[buffer_name] = DataBuffer(max_size)
            self.metrics[buffer_name] = FlowMetrics()
    
    def _initialize_default_routes(self):
        """Initialize default data flow routes."""
        # Training data flow
        self.add_route(FlowRoute(
            name="raw_to_training",
            source="data_loader",
            destination="training_pipeline",
            data_type=DataType.RAW_TEXT,
            priority=10
        ))
        
        # Inference data flow
        self.add_route(FlowRoute(
            name="inference_to_evaluation",
            source="inference_engine",
            destination="evaluation_pipeline",
            data_type=DataType.MODEL_PREDICTIONS,
            priority=8
        ))
        
        # Metrics flow
        self.add_route(FlowRoute(
            name="metrics_to_dashboard",
            source="metrics_computer",
            destination="dashboard",
            data_type=DataType.METRICS,
            priority=5
        ))
        
        # Events flow
        self.add_route(FlowRoute(
            name="events_to_monitoring",
            source="event_bus",
            destination="monitoring_system",
            data_type=DataType.EVENTS,
            priority=3
        ))
    
    def start(self) -> bool:
        """
        Start data flow coordination.
        
        Returns:
            True if started successfully
        """
        try:
            with self._lock:
                if self.worker_running:
                    return True
                
                self.worker_running = True
                self.state = FlowState.ACTIVE
                
                # Start worker threads for each buffer
                for buffer_name in self.buffers:
                    worker = threading.Thread(
                        target=self._buffer_worker,
                        args=(buffer_name,),
                        name=f"DataFlow-{buffer_name}"
                    )
                    worker.daemon = True
                    worker.start()
                    self.workers[buffer_name] = worker
                
                # Start metrics collection
                metrics_worker = threading.Thread(
                    target=self._metrics_worker,
                    name="DataFlow-Metrics"
                )
                metrics_worker.daemon = True
                metrics_worker.start()
                self.workers['metrics'] = metrics_worker
                
                logger.info("Data flow coordinator started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start data flow coordinator: {e}")
            return False
    
    def stop(self):
        """Stop data flow coordination."""
        try:
            with self._lock:
                self.worker_running = False
                self.state = FlowState.STOPPED
                
                # Wait for workers to finish
                for worker in self.workers.values():
                    if worker.is_alive():
                        worker.join(timeout=5)
                
                self.workers.clear()
                
                # Clear buffers
                for buffer in self.buffers.values():
                    buffer.clear()
                
                logger.info("Data flow coordinator stopped")
                
        except Exception as e:
            logger.error(f"Error stopping data flow coordinator: {e}")
    
    def register_component(self, name: str, component: Any) -> bool:
        """
        Register component for data flow.
        
        Args:
            name: Component name
            component: Component instance
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                self.components[name] = component
                
                # Create buffer if needed
                if name not in self.buffers:
                    self.buffers[name] = DataBuffer()
                    self.metrics[name] = FlowMetrics()
            
            logger.info(f"Registered component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {name}: {e}")
            return False
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister component.
        
        Args:
            name: Component name
            
        Returns:
            True if unregistered successfully
        """
        try:
            with self._lock:
                self.components.pop(name, None)
                
                # Clear and remove buffer
                if name in self.buffers:
                    self.buffers[name].clear()
                    del self.buffers[name]
                
                self.metrics.pop(name, None)
            
            logger.info(f"Unregistered component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister component {name}: {e}")
            return False
    
    def add_route(self, route: FlowRoute) -> bool:
        """
        Add data flow route.
        
        Args:
            route: Flow route definition
            
        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                self.routes[route.name] = route
                
                # Create buffers for source and destination if needed
                for component in [route.source, route.destination]:
                    if component and component not in self.buffers:
                        self.buffers[component] = DataBuffer()
                        self.metrics[component] = FlowMetrics()
            
            logger.info(f"Added flow route: {route.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add route {route.name}: {e}")
            return False
    
    def remove_route(self, route_name: str) -> bool:
        """
        Remove data flow route.
        
        Args:
            route_name: Route name
            
        Returns:
            True if removed successfully
        """
        try:
            with self._lock:
                if route_name in self.routes:
                    del self.routes[route_name]
                    logger.info(f"Removed flow route: {route_name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove route {route_name}: {e}")
            return False
    
    def send_data(self, 
                 data: Any,
                 data_type: DataType,
                 source: str,
                 destination: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 priority: int = 0) -> bool:
        """
        Send data through flow system.
        
        Args:
            data: Data to send
            data_type: Type of data
            source: Source component
            destination: Destination component (optional)
            metadata: Additional metadata
            priority: Message priority
            
        Returns:
            True if sent successfully
        """
        try:
            # Create data packet
            packet = DataPacket(
                data=data,
                data_type=data_type,
                source=source,
                destination=destination,
                metadata=metadata or {},
                priority=priority
            )
            
            # Find appropriate routes
            routes = self._find_routes(packet)
            
            if not routes:
                logger.warning(f"No routes found for packet from {source} to {destination}")
                return False
            
            # Send through each route
            success = False
            for route in routes:
                if self._send_through_route(packet, route):
                    success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            return False
    
    def _find_routes(self, packet: DataPacket) -> List[FlowRoute]:
        """Find applicable routes for packet."""
        routes = []
        
        with self._lock:
            for route in self.routes.values():
                if not route.enabled:
                    continue
                
                # Check source
                if route.source != packet.source:
                    continue
                
                # Check destination if specified
                if packet.destination and route.destination != packet.destination:
                    continue
                
                # Check data type
                if route.data_type != packet.data_type:
                    continue
                
                # Apply filter if present
                if route.filter_fn and not route.filter_fn(packet):
                    continue
                
                routes.append(route)
        
        # Sort by priority
        routes.sort(key=lambda r: r.priority, reverse=True)
        
        return routes
    
    def _send_through_route(self, packet: DataPacket, route: FlowRoute) -> bool:
        """Send packet through specific route."""
        try:
            # Apply transformation if present
            if route.transform_fn:
                packet = route.transform_fn(packet)
            
            # Apply processor if present
            if route.processor:
                processed_data = route.processor.process(packet.data)
                packet.data = processed_data
            
            # Get destination buffer
            destination_buffer = self.buffers.get(route.destination)
            if not destination_buffer:
                logger.error(f"No buffer for destination: {route.destination}")
                return False
            
            # Send to buffer
            success = destination_buffer.put(packet)
            
            if success:
                # Update metrics
                self._update_flow_metrics(route.destination, packet)
                
                # Publish event
                self._publish_flow_event(packet, route)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send through route {route.name}: {e}")
            return False
    
    def _buffer_worker(self, buffer_name: str):
        """Worker thread for processing buffer."""
        buffer = self.buffers[buffer_name]
        
        while self.worker_running:
            try:
                # Get packet from buffer
                packet = buffer.get(timeout=1.0)
                
                if packet:
                    # Process packet
                    self._process_packet(packet, buffer_name)
                
            except Exception as e:
                logger.error(f"Buffer worker error for {buffer_name}: {e}")
    
    def _process_packet(self, packet: DataPacket, buffer_name: str):
        """Process individual packet."""
        try:
            # Get component
            component = self.components.get(buffer_name)
            
            if component and hasattr(component, 'process_data'):
                # Component has data processing method
                component.process_data(packet)
            
            # Update metrics
            metrics = self.metrics[buffer_name]
            metrics.last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to process packet in {buffer_name}: {e}")
            
            # Update error metrics
            metrics = self.metrics[buffer_name]
            metrics.packets_failed += 1
    
    def _metrics_worker(self):
        """Worker thread for metrics collection."""
        while self.worker_running:
            try:
                # Collect and update metrics
                for buffer_name, buffer in self.buffers.items():
                    buffer_stats = buffer.get_statistics()
                    metrics = self.metrics[buffer_name]
                    
                    # Update throughput
                    current_time = datetime.now()
                    if metrics.last_activity:
                        time_diff = (current_time - metrics.last_activity).total_seconds()
                        if time_diff > 0:
                            metrics.current_throughput = buffer_stats['total_removed'] / time_diff
                    
                    # Update error rate
                    total_packets = buffer_stats['total_added']
                    if total_packets > 0:
                        metrics.error_rate = metrics.packets_failed / total_packets
                
                # Sleep for metrics interval
                import time
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
    
    def _update_flow_metrics(self, destination: str, packet: DataPacket):
        """Update flow metrics."""
        metrics = self.metrics.get(destination)
        
        if metrics:
            metrics.packets_processed += 1
            metrics.total_bytes += len(str(packet.data)) if packet.data else 0
            metrics.last_activity = datetime.now()
            
            # Update latency
            if 'send_time' in packet.metadata:
                send_time = datetime.fromisoformat(packet.metadata['send_time'])
                latency = (datetime.now() - send_time).total_seconds()
                
                # Exponential moving average
                if metrics.average_latency == 0:
                    metrics.average_latency = latency
                else:
                    metrics.average_latency = 0.9 * metrics.average_latency + 0.1 * latency
    
    def _publish_flow_event(self, packet: DataPacket, route: FlowRoute):
        """Publish data flow event."""
        event = Event(
            type=EventType.DATA_FLOW,
            data={
                'packet_id': packet.packet_id,
                'route_name': route.name,
                'source': packet.source,
                'destination': packet.destination,
                'data_type': packet.data_type.value,
                'timestamp': packet.timestamp.isoformat()
            },
            source="DataFlowCoordinator"
        )
        
        self.event_bus.publish(event)
    
    def get_buffer_status(self, buffer_name: str) -> Optional[Dict[str, Any]]:
        """
        Get buffer status.
        
        Args:
            buffer_name: Buffer name
            
        Returns:
            Buffer status or None
        """
        buffer = self.buffers.get(buffer_name)
        metrics = self.metrics.get(buffer_name)
        
        if buffer and metrics:
            return {
                'buffer_stats': buffer.get_statistics(),
                'flow_metrics': {
                    'packets_processed': metrics.packets_processed,
                    'packets_failed': metrics.packets_failed,
                    'total_bytes': metrics.total_bytes,
                    'average_latency': metrics.average_latency,
                    'current_throughput': metrics.current_throughput,
                    'error_rate': metrics.error_rate,
                    'last_activity': metrics.last_activity.isoformat() if metrics.last_activity else None
                }
            }
        
        return None
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get comprehensive flow statistics."""
        with self._lock:
            stats = {
                'state': self.state.value,
                'total_routes': len(self.routes),
                'active_routes': len([r for r in self.routes.values() if r.enabled]),
                'total_components': len(self.components),
                'total_buffers': len(self.buffers),
                'worker_running': self.worker_running,
                'buffer_stats': {},
                'route_stats': {}
            }
            
            # Buffer statistics
            for name, buffer in self.buffers.items():
                stats['buffer_stats'][name] = self.get_buffer_status(name)
            
            # Route statistics
            for name, route in self.routes.items():
                stats['route_stats'][name] = {
                    'enabled': route.enabled,
                    'priority': route.priority,
                    'source': route.source,
                    'destination': route.destination,
                    'data_type': route.data_type.value
                }
            
            return stats
    
    def pause_flow(self, component: Optional[str] = None) -> bool:
        """
        Pause data flow.
        
        Args:
            component: Specific component to pause (all if None)
            
        Returns:
            True if paused successfully
        """
        try:
            with self._lock:
                if component:
                    # Pause specific component routes
                    for route in self.routes.values():
                        if route.source == component or route.destination == component:
                            route.enabled = False
                else:
                    # Pause all flows
                    self.state = FlowState.PAUSED
                    for route in self.routes.values():
                        route.enabled = False
            
            logger.info(f"Paused data flow: {component or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause flow: {e}")
            return False
    
    def resume_flow(self, component: Optional[str] = None) -> bool:
        """
        Resume data flow.
        
        Args:
            component: Specific component to resume (all if None)
            
        Returns:
            True if resumed successfully
        """
        try:
            with self._lock:
                if component:
                    # Resume specific component routes
                    for route in self.routes.values():
                        if route.source == component or route.destination == component:
                            route.enabled = True
                else:
                    # Resume all flows
                    self.state = FlowState.ACTIVE
                    for route in self.routes.values():
                        route.enabled = True
            
            logger.info(f"Resumed data flow: {component or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume flow: {e}")
            return False


# Global coordinator instance
_data_coordinator = None

def get_data_coordinator() -> DataFlowCoordinator:
    """Get global data flow coordinator instance."""
    global _data_coordinator
    if _data_coordinator is None:
        _data_coordinator = DataFlowCoordinator()
    return _data_coordinator


# Convenience functions

def send_training_data(data: Any, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send training data through flow system.
    
    Args:
        data: Training data
        source: Source component
        metadata: Additional metadata
        
    Returns:
        True if sent successfully
    """
    coordinator = get_data_coordinator()
    return coordinator.send_data(
        data=data,
        data_type=DataType.TRAINING_DATA,
        source=source,
        metadata=metadata,
        priority=10
    )


def send_metrics(metrics: Dict[str, Any], source: str) -> bool:
    """
    Send metrics through flow system.
    
    Args:
        metrics: Metrics data
        source: Source component
        
    Returns:
        True if sent successfully
    """
    coordinator = get_data_coordinator()
    return coordinator.send_data(
        data=metrics,
        data_type=DataType.METRICS,
        source=source,
        destination="dashboard",
        priority=5
    )


def register_data_component(name: str, component: Any) -> bool:
    """
    Register component for data flow.
    
    Args:
        name: Component name
        component: Component instance
        
    Returns:
        True if registered successfully
    """
    coordinator = get_data_coordinator()
    return coordinator.register_component(name, component)