"""
Event system for fine-tune LLM library.

Provides publish/subscribe messaging, event storage, and analytics
for real-time communication between components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Standard event types."""
    
    # Training events
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    EPOCH_STARTED = "training.epoch.started"
    EPOCH_COMPLETED = "training.epoch.completed"
    BATCH_COMPLETED = "training.batch.completed"
    METRICS_UPDATED = "training.metrics.updated"
    CHECKPOINT_SAVED = "training.checkpoint.saved"
    
    # Model events
    MODEL_LOADED = "model.loaded"
    MODEL_SAVED = "model.saved"
    MODEL_VALIDATED = "model.validated"
    ADAPTER_APPLIED = "model.adapter.applied"
    ADAPTER_REMOVED = "model.adapter.removed"
    
    # Inference events
    PREDICTION_STARTED = "inference.prediction.started"
    PREDICTION_COMPLETED = "inference.prediction.completed"
    CALIBRATION_COMPUTED = "inference.calibration.computed"
    UNCERTAINTY_COMPUTED = "inference.uncertainty.computed"
    CONFORMAL_PREDICTION = "inference.conformal.prediction"
    
    # System events
    SERVICE_STARTED = "system.service.started"
    SERVICE_STOPPED = "system.service.stopped"
    ERROR_OCCURRED = "system.error.occurred"
    HEALTH_CHECK = "system.health.check"
    RESOURCE_WARNING = "system.resource.warning"
    
    # User events
    UI_ACTION = "user.ui.action"
    CONFIG_CHANGED = "user.config.changed"
    DASHBOARD_VIEWED = "user.dashboard.viewed"
    
    # Data events
    DATA_LOADED = "data.loaded"
    DATA_PROCESSED = "data.processed"
    DATA_VALIDATED = "data.validated"
    DATASET_SPLIT = "data.dataset.split"

@dataclass
class Event:
    """Base event class."""
    
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            source=data["source"],
            data=data["data"],
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id")
        )
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))

class EventHandler(ABC):
    """Abstract event handler."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if handler can process event type."""
        pass

class EventBus:
    """Central event bus for publish/subscribe messaging."""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.middleware: List[Callable] = []
        self.running = False
        self.event_queue = asyncio.Queue()
        self._stats = {
            "events_published": 0,
            "events_handled": 0,
            "handler_errors": 0
        }
    
    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Register event handler."""
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler {handler.__class__.__name__} for {event_type.value}")
    
    def unregister_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """Unregister event handler."""
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            logger.info(f"Unregistered handler {handler.__class__.__name__} for {event_type.value}")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to event type with callback."""
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed callback {callback.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe callback from event type."""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.info(f"Unsubscribed callback {callback.__name__} from {event_type.value}")
    
    def add_middleware(self, middleware: Callable[[Event], Event]) -> None:
        """Add middleware to process events before handling."""
        self.middleware.append(middleware)
    
    async def publish(self, event: Event) -> None:
        """Publish event to bus."""
        self._stats["events_published"] += 1
        
        # Apply middleware
        for middleware_func in self.middleware:
            try:
                event = middleware_func(event)
            except Exception as e:
                logger.error(f"Middleware error: {e}")
        
        await self.event_queue.put(event)
    
    async def start(self) -> None:
        """Start event bus processing."""
        self.running = True
        logger.info("Event bus started")
        
        while self.running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def stop(self) -> None:
        """Stop event bus processing."""
        self.running = False
        logger.info("Event bus stopped")
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        try:
            # Handle with registered handlers
            handlers = self.handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if handler.can_handle(event.event_type):
                        await handler.handle(event)
                        self._stats["events_handled"] += 1
                except Exception as e:
                    self._stats["handler_errors"] += 1
                    logger.error(f"Handler error: {e}")
            
            # Call subscribers
            subscribers = self.subscribers.get(event.event_type, [])
            for callback in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                    self._stats["events_handled"] += 1
                except Exception as e:
                    self._stats["handler_errors"] += 1
                    logger.error(f"Subscriber error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing event {event.event_type}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return self._stats.copy()

class EventStore:
    """Event storage for persistence and replay."""
    
    def __init__(self, storage_path: Optional[Path] = None, max_events: int = 10000):
        self.storage_path = storage_path
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.event_index: Dict[EventType, List[Event]] = defaultdict(list)
    
    def store(self, event: Event) -> None:
        """Store event."""
        self.events.append(event)
        self.event_index[event.event_type].append(event)
        
        # Persist if storage path provided
        if self.storage_path:
            self._persist_event(event)
    
    def get_events(self, event_type: Optional[EventType] = None, 
                  since: Optional[float] = None, limit: Optional[int] = None) -> List[Event]:
        """Get events with optional filtering."""
        if event_type:
            events = self.event_index[event_type]
        else:
            events = list(self.events)
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_event_count(self, event_type: Optional[EventType] = None) -> int:
        """Get count of events."""
        if event_type:
            return len(self.event_index[event_type])
        return len(self.events)
    
    def clear(self) -> None:
        """Clear all stored events."""
        self.events.clear()
        self.event_index.clear()
    
    def _persist_event(self, event: Event) -> None:
        """Persist event to storage."""
        try:
            if not self.storage_path.exists():
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'a') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

class EventAggregator:
    """Event aggregation and analytics."""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    def get_event_counts_by_type(self, since: Optional[float] = None) -> Dict[str, int]:
        """Get event counts grouped by type."""
        counts = {}
        for event_type in EventType:
            events = self.event_store.get_events(event_type, since=since)
            counts[event_type.value] = len(events)
        return counts
    
    def get_event_timeline(self, event_type: EventType, 
                          window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get event timeline with time windows."""
        since = time.time() - (window_minutes * 60)
        events = self.event_store.get_events(event_type, since=since)
        
        # Group events by time windows
        window_size = window_minutes * 60 / 10  # 10 windows
        timeline = []
        
        for i in range(10):
            window_start = since + (i * window_size)
            window_end = since + ((i + 1) * window_size)
            
            window_events = [
                e for e in events 
                if window_start <= e.timestamp < window_end
            ]
            
            timeline.append({
                "window_start": window_start,
                "window_end": window_end,
                "event_count": len(window_events),
                "events": [e.to_dict() for e in window_events]
            })
        
        return timeline
    
    def get_error_summary(self, since: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of error events."""
        error_events = self.event_store.get_events(EventType.ERROR_OCCURRED, since=since)
        
        error_counts = defaultdict(int)
        error_sources = defaultdict(int)
        
        for event in error_events:
            error_type = event.data.get("error_type", "unknown")
            error_counts[error_type] += 1
            error_sources[event.source] += 1
        
        return {
            "total_errors": len(error_events),
            "error_types": dict(error_counts),
            "error_sources": dict(error_sources),
            "recent_errors": [e.to_dict() for e in error_events[-10:]]
        }

# Utility functions for creating common events
def create_training_event(event_type: EventType, source: str, 
                         epoch: Optional[int] = None, batch: Optional[int] = None,
                         metrics: Optional[Dict[str, float]] = None,
                         **kwargs) -> Event:
    """Create training-related event."""
    data = kwargs.copy()
    if epoch is not None:
        data["epoch"] = epoch
    if batch is not None:
        data["batch"] = batch
    if metrics:
        data["metrics"] = metrics
    
    return Event(event_type=event_type, source=source, data=data)

def create_model_event(event_type: EventType, source: str,
                      model_path: Optional[str] = None, model_type: Optional[str] = None,
                      **kwargs) -> Event:
    """Create model-related event."""
    data = kwargs.copy()
    if model_path:
        data["model_path"] = model_path
    if model_type:
        data["model_type"] = model_type
    
    return Event(event_type=event_type, source=source, data=data)

def create_error_event(source: str, error: Exception, **kwargs) -> Event:
    """Create error event."""
    data = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "error_traceback": str(error.__traceback__) if error.__traceback__ else None,
        **kwargs
    }
    
    return Event(event_type=EventType.ERROR_OCCURRED, source=source, data=data)

# Global event bus instance
_global_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus

# Export all event components
__all__ = [
    "EventType",
    "Event", 
    "EventHandler",
    "EventBus",
    "EventStore",
    "EventAggregator",
    "create_training_event",
    "create_model_event", 
    "create_error_event",
    "get_event_bus"
]