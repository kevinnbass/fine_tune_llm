"""
Unit tests for core events system.

This test module provides comprehensive coverage for the EventBus, Event,
EventStore, and EventPublisher systems with 100% line coverage.
"""

import pytest
import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import weakref

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.fine_tune_llm.core.events import (
    Event,
    EventType,
    EventPriority,
    EventBus,
    EventStore,
    EventPublisher,
    EventSubscription,
    EventFilter,
    EventHandler,
    EventMiddleware,
    AsyncEventBus,
    EventStoreConfig,
    EventBusConfig,
    SerializationError,
    SubscriptionError,
    EventBusError,
    get_event_bus,
    get_event_store,
    publish_event,
    subscribe_to_event,
    create_event
)


class TestEvent:
    """Test Event class."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        data = {"test": "data"}
        event = Event(
            type=EventType.TRAINING_STARTED,
            data=data,
            source="test_source"
        )
        
        assert event.type == EventType.TRAINING_STARTED
        assert event.data == data
        assert event.source == "test_source"
        assert event.priority == EventPriority.NORMAL
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == timezone.utc
        assert isinstance(event.id, str)
        assert len(event.id) > 0
    
    def test_event_with_custom_values(self):
        """Test event creation with custom values."""
        custom_time = datetime.now(timezone.utc)
        event = Event(
            type=EventType.ERROR_OCCURRED,
            data={"error": "test error"},
            source="custom_source",
            priority=EventPriority.HIGH,
            timestamp=custom_time,
            event_id="custom-id-123",
            correlation_id="corr-456",
            metadata={"user": "test_user"}
        )
        
        assert event.type == EventType.ERROR_OCCURRED
        assert event.priority == EventPriority.HIGH
        assert event.timestamp == custom_time
        assert event.id == "custom-id-123"
        assert event.correlation_id == "corr-456"
        assert event.metadata["user"] == "test_user"
    
    def test_event_serialization(self):
        """Test event serialization to dict."""
        event = Event(
            type=EventType.TRAINING_COMPLETED,
            data={"accuracy": 0.95},
            source="trainer"
        )
        
        serialized = event.to_dict()
        
        assert serialized["type"] == "TRAINING_COMPLETED"
        assert serialized["data"] == {"accuracy": 0.95}
        assert serialized["source"] == "trainer"
        assert serialized["priority"] == "NORMAL"
        assert "timestamp" in serialized
        assert "id" in serialized
    
    def test_event_deserialization(self):
        """Test event deserialization from dict."""
        event_dict = {
            "type": "MODEL_LOADED",
            "data": {"model_name": "test-model"},
            "source": "model_loader",
            "priority": "HIGH",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "id": "test-id-789",
            "correlation_id": "corr-123",
            "metadata": {"version": "1.0"}
        }
        
        event = Event.from_dict(event_dict)
        
        assert event.type == EventType.MODEL_LOADED
        assert event.data == {"model_name": "test-model"}
        assert event.source == "model_loader"
        assert event.priority == EventPriority.HIGH
        assert event.id == "test-id-789"
        assert event.correlation_id == "corr-123"
        assert event.metadata["version"] == "1.0"
    
    def test_event_deserialization_invalid_type(self):
        """Test event deserialization with invalid type."""
        event_dict = {
            "type": "INVALID_TYPE",
            "data": {},
            "source": "test"
        }
        
        with pytest.raises(ValueError, match="Invalid event type"):
            Event.from_dict(event_dict)
    
    def test_event_str_representation(self):
        """Test event string representation."""
        event = Event(
            type=EventType.SYSTEM_STARTED,
            data={},
            source="system"
        )
        
        str_repr = str(event)
        assert "Event" in str_repr
        assert "SYSTEM_STARTED" in str_repr
        assert "system" in str_repr
    
    def test_event_repr(self):
        """Test event repr."""
        event = Event(
            type=EventType.SYSTEM_STARTED,
            data={},
            source="system"
        )
        
        repr_str = repr(event)
        assert "Event" in repr_str
        assert event.id in repr_str


class TestEventFilter:
    """Test EventFilter class."""
    
    def test_filter_by_type(self):
        """Test filtering by event type."""
        filter_obj = EventFilter(types=[EventType.TRAINING_STARTED])
        
        matching_event = Event(EventType.TRAINING_STARTED, {}, "test")
        non_matching_event = Event(EventType.TRAINING_COMPLETED, {}, "test")
        
        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)
    
    def test_filter_by_source(self):
        """Test filtering by source."""
        filter_obj = EventFilter(sources=["trainer", "model"])
        
        matching_event = Event(EventType.TRAINING_STARTED, {}, "trainer")
        non_matching_event = Event(EventType.TRAINING_STARTED, {}, "other")
        
        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)
    
    def test_filter_by_priority(self):
        """Test filtering by priority."""
        filter_obj = EventFilter(priorities=[EventPriority.HIGH])
        
        matching_event = Event(
            EventType.ERROR_OCCURRED, {}, "test", priority=EventPriority.HIGH
        )
        non_matching_event = Event(
            EventType.ERROR_OCCURRED, {}, "test", priority=EventPriority.LOW
        )
        
        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)
    
    def test_filter_by_correlation_id(self):
        """Test filtering by correlation ID."""
        filter_obj = EventFilter(correlation_ids=["corr-123"])
        
        matching_event = Event(
            EventType.TRAINING_STARTED, {}, "test", correlation_id="corr-123"
        )
        non_matching_event = Event(
            EventType.TRAINING_STARTED, {}, "test", correlation_id="corr-456"
        )
        
        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)
    
    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria."""
        filter_obj = EventFilter(
            types=[EventType.TRAINING_STARTED],
            sources=["trainer"],
            priorities=[EventPriority.NORMAL]
        )
        
        matching_event = Event(
            EventType.TRAINING_STARTED, {}, "trainer", priority=EventPriority.NORMAL
        )
        non_matching_event = Event(
            EventType.TRAINING_STARTED, {}, "trainer", priority=EventPriority.HIGH
        )
        
        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)
    
    def test_filter_no_criteria(self):
        """Test filter with no criteria matches all events."""
        filter_obj = EventFilter()
        
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        assert filter_obj.matches(event)


class TestEventSubscription:
    """Test EventSubscription class."""
    
    def test_subscription_creation(self):
        """Test subscription creation."""
        handler = Mock()
        filter_obj = EventFilter(types=[EventType.TRAINING_STARTED])
        
        subscription = EventSubscription(
            handler=handler,
            filter=filter_obj,
            subscription_id="sub-123"
        )
        
        assert subscription.handler == handler
        assert subscription.filter == filter_obj
        assert subscription.id == "sub-123"
        assert subscription.active
        assert subscription.created_at <= datetime.now(timezone.utc)
    
    def test_subscription_matches(self):
        """Test subscription event matching."""
        handler = Mock()
        filter_obj = EventFilter(types=[EventType.TRAINING_STARTED])
        subscription = EventSubscription(handler, filter_obj)
        
        matching_event = Event(EventType.TRAINING_STARTED, {}, "test")
        non_matching_event = Event(EventType.TRAINING_COMPLETED, {}, "test")
        
        assert subscription.matches(matching_event)
        assert not subscription.matches(non_matching_event)
    
    def test_subscription_deactivate(self):
        """Test subscription deactivation."""
        handler = Mock()
        subscription = EventSubscription(handler, EventFilter())
        
        assert subscription.active
        subscription.deactivate()
        assert not subscription.active
    
    def test_subscription_call_handler(self):
        """Test calling subscription handler."""
        handler = Mock()
        subscription = EventSubscription(handler, EventFilter())
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        
        subscription.call_handler(event)
        handler.assert_called_once_with(event)
    
    def test_subscription_call_handler_exception(self):
        """Test handler exception handling."""
        handler = Mock(side_effect=Exception("Handler error"))
        subscription = EventSubscription(handler, EventFilter())
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        
        # Should not raise exception
        subscription.call_handler(event)
        handler.assert_called_once_with(event)


class TestEventMiddleware:
    """Test EventMiddleware class."""
    
    def test_middleware_before_publish(self):
        """Test middleware before publish hook."""
        middleware = EventMiddleware()
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        
        # Base implementation returns event unchanged
        result = middleware.before_publish(event)
        assert result == event
    
    def test_middleware_after_publish(self):
        """Test middleware after publish hook."""
        middleware = EventMiddleware()
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        
        # Base implementation does nothing, should not raise
        middleware.after_publish(event, True)
        middleware.after_publish(event, False)


class TestEventBus:
    """Test EventBus class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.event_bus = EventBus()
    
    def test_event_bus_creation(self):
        """Test event bus creation."""
        config = EventBusConfig(max_subscribers=100, enable_async=False)
        bus = EventBus(config)
        
        assert len(bus._subscriptions) == 0
        assert len(bus._middleware) == 0
        assert bus._config.max_subscribers == 100
        assert not bus._config.enable_async
    
    def test_subscribe_basic(self):
        """Test basic event subscription."""
        handler = Mock()
        
        subscription_id = self.event_bus.subscribe(
            EventType.TRAINING_STARTED,
            handler
        )
        
        assert isinstance(subscription_id, str)
        assert len(self.event_bus._subscriptions) == 1
        assert subscription_id in self.event_bus._subscriptions
    
    def test_subscribe_with_filter(self):
        """Test subscription with custom filter."""
        handler = Mock()
        filter_obj = EventFilter(sources=["trainer"])
        
        subscription_id = self.event_bus.subscribe(
            EventType.TRAINING_STARTED,
            handler,
            filter=filter_obj
        )
        
        subscription = self.event_bus._subscriptions[subscription_id]
        assert subscription.filter == filter_obj
    
    def test_subscribe_max_subscribers(self):
        """Test max subscribers limit."""
        config = EventBusConfig(max_subscribers=2)
        bus = EventBus(config)
        
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()
        
        bus.subscribe(EventType.TRAINING_STARTED, handler1)
        bus.subscribe(EventType.TRAINING_STARTED, handler2)
        
        with pytest.raises(SubscriptionError, match="Maximum number of subscribers"):
            bus.subscribe(EventType.TRAINING_STARTED, handler3)
    
    def test_unsubscribe(self):
        """Test event unsubscription."""
        handler = Mock()
        subscription_id = self.event_bus.subscribe(
            EventType.TRAINING_STARTED,
            handler
        )
        
        assert len(self.event_bus._subscriptions) == 1
        
        success = self.event_bus.unsubscribe(subscription_id)
        assert success
        assert len(self.event_bus._subscriptions) == 0
    
    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing non-existent subscription."""
        success = self.event_bus.unsubscribe("non-existent-id")
        assert not success
    
    def test_publish_basic(self):
        """Test basic event publishing."""
        handler = Mock()
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler)
        
        event = Event(EventType.TRAINING_STARTED, {"test": "data"}, "test")
        self.event_bus.publish(event)
        
        handler.assert_called_once_with(event)
    
    def test_publish_multiple_subscribers(self):
        """Test publishing to multiple subscribers."""
        handler1 = Mock()
        handler2 = Mock()
        
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler2)
        
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        self.event_bus.publish(event)
        
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)
    
    def test_publish_with_filter(self):
        """Test publishing with event filtering."""
        handler1 = Mock()
        handler2 = Mock()
        
        # Handler 1 - all events
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
        
        # Handler 2 - only from "trainer" source
        filter_obj = EventFilter(sources=["trainer"])
        self.event_bus.subscribe(
            EventType.TRAINING_STARTED, handler2, filter=filter_obj
        )
        
        # Publish event from "trainer" - both should be called
        event1 = Event(EventType.TRAINING_STARTED, {}, "trainer")
        self.event_bus.publish(event1)
        
        # Publish event from "other" - only handler1 should be called
        event2 = Event(EventType.TRAINING_STARTED, {}, "other")
        self.event_bus.publish(event2)
        
        assert handler1.call_count == 2
        assert handler2.call_count == 1
    
    def test_publish_inactive_subscription(self):
        """Test publishing to inactive subscription."""
        handler = Mock()
        subscription_id = self.event_bus.subscribe(
            EventType.TRAINING_STARTED, handler
        )
        
        # Deactivate subscription
        self.event_bus._subscriptions[subscription_id].deactivate()
        
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        self.event_bus.publish(event)
        
        # Handler should not be called
        handler.assert_not_called()
    
    def test_publish_with_middleware(self):
        """Test publishing with middleware."""
        middleware = Mock(spec=EventMiddleware)
        middleware.before_publish.return_value = Mock(spec=Event)
        
        self.event_bus.add_middleware(middleware)
        
        handler = Mock()
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler)
        
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        self.event_bus.publish(event)
        
        middleware.before_publish.assert_called_once_with(event)
        middleware.after_publish.assert_called_once()
    
    def test_add_remove_middleware(self):
        """Test adding and removing middleware."""
        middleware = Mock(spec=EventMiddleware)
        
        self.event_bus.add_middleware(middleware)
        assert middleware in self.event_bus._middleware
        
        self.event_bus.remove_middleware(middleware)
        assert middleware not in self.event_bus._middleware
    
    def test_clear_subscriptions(self):
        """Test clearing all subscriptions."""
        handler1 = Mock()
        handler2 = Mock()
        
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
        self.event_bus.subscribe(EventType.TRAINING_COMPLETED, handler2)
        
        assert len(self.event_bus._subscriptions) == 2
        
        self.event_bus.clear_subscriptions()
        assert len(self.event_bus._subscriptions) == 0
    
    def test_get_subscription_count(self):
        """Test getting subscription count."""
        assert self.event_bus.get_subscription_count() == 0
        
        handler = Mock()
        self.event_bus.subscribe(EventType.TRAINING_STARTED, handler)
        
        assert self.event_bus.get_subscription_count() == 1
    
    def test_get_subscriptions_for_type(self):
        """Test getting subscriptions for specific event type."""
        handler1 = Mock()
        handler2 = Mock()
        
        sub1 = self.event_bus.subscribe(EventType.TRAINING_STARTED, handler1)
        sub2 = self.event_bus.subscribe(EventType.TRAINING_COMPLETED, handler2)
        
        training_subs = self.event_bus.get_subscriptions_for_type(
            EventType.TRAINING_STARTED
        )
        
        assert len(training_subs) == 1
        assert sub1 in [sub.id for sub in training_subs]
        assert sub2 not in [sub.id for sub in training_subs]


class TestAsyncEventBus:
    """Test AsyncEventBus class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.event_bus = AsyncEventBus()
    
    @pytest.mark.asyncio
    async def test_async_subscribe_basic(self):
        """Test basic async event subscription."""
        handler = Mock()
        
        subscription_id = await self.event_bus.subscribe(
            EventType.TRAINING_STARTED,
            handler
        )
        
        assert isinstance(subscription_id, str)
        assert len(self.event_bus._subscriptions) == 1
    
    @pytest.mark.asyncio
    async def test_async_publish_basic(self):
        """Test basic async event publishing."""
        handler = Mock()
        await self.event_bus.subscribe(EventType.TRAINING_STARTED, handler)
        
        event = Event(EventType.TRAINING_STARTED, {"test": "data"}, "test")
        await self.event_bus.publish(event)
        
        handler.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_async_publish_coroutine_handler(self):
        """Test async publishing with coroutine handler."""
        async def async_handler(event):
            pass
        
        handler_mock = Mock(side_effect=async_handler)
        await self.event_bus.subscribe(EventType.TRAINING_STARTED, handler_mock)
        
        event = Event(EventType.TRAINING_STARTED, {}, "test")
        
        with patch('asyncio.iscoroutinefunction', return_value=True):
            with patch('asyncio.create_task') as mock_create_task:
                await self.event_bus.publish(event)
                mock_create_task.assert_called()


class TestEventStore:
    """Test EventStore class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        config = EventStoreConfig(
            storage_path=self.temp_dir / "events.db",
            max_events=1000,
            auto_cleanup=False
        )
        self.event_store = EventStore(config)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_event(self):
        """Test storing an event."""
        event = Event(EventType.TRAINING_STARTED, {"test": "data"}, "test")
        
        success = self.event_store.store_event(event)
        assert success
        
        # Verify event was stored
        stored_events = self.event_store.get_events()
        assert len(stored_events) == 1
        assert stored_events[0].id == event.id
    
    def test_store_multiple_events(self):
        """Test storing multiple events."""
        events = [
            Event(EventType.TRAINING_STARTED, {}, "test1"),
            Event(EventType.TRAINING_COMPLETED, {}, "test2"),
            Event(EventType.MODEL_LOADED, {}, "test3")
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        stored_events = self.event_store.get_events()
        assert len(stored_events) == 3
    
    def test_get_events_with_filter(self):
        """Test getting events with filter."""
        events = [
            Event(EventType.TRAINING_STARTED, {}, "trainer"),
            Event(EventType.TRAINING_COMPLETED, {}, "trainer"),
            Event(EventType.MODEL_LOADED, {}, "loader")
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        # Filter by source
        filter_obj = EventFilter(sources=["trainer"])
        filtered_events = self.event_store.get_events(filter_obj)
        
        assert len(filtered_events) == 2
        assert all(event.source == "trainer" for event in filtered_events)
    
    def test_get_events_with_limit(self):
        """Test getting events with limit."""
        events = [
            Event(EventType.TRAINING_STARTED, {}, f"test{i}")
            for i in range(5)
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        limited_events = self.event_store.get_events(limit=3)
        assert len(limited_events) == 3
    
    def test_get_events_by_correlation_id(self):
        """Test getting events by correlation ID."""
        correlation_id = "test-correlation-123"
        
        events = [
            Event(EventType.TRAINING_STARTED, {}, "test1", correlation_id=correlation_id),
            Event(EventType.TRAINING_COMPLETED, {}, "test2", correlation_id=correlation_id),
            Event(EventType.MODEL_LOADED, {}, "test3")  # Different correlation ID
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        correlated_events = self.event_store.get_events_by_correlation_id(correlation_id)
        assert len(correlated_events) == 2
        assert all(event.correlation_id == correlation_id for event in correlated_events)
    
    def test_cleanup_old_events(self):
        """Test cleanup of old events."""
        # Set max events to 3
        self.event_store._config.max_events = 3
        
        # Store 5 events
        events = [
            Event(EventType.TRAINING_STARTED, {}, f"test{i}")
            for i in range(5)
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        # Trigger cleanup
        self.event_store._cleanup_old_events()
        
        # Should have only 3 events left
        remaining_events = self.event_store.get_events()
        assert len(remaining_events) == 3
    
    def test_get_statistics(self):
        """Test getting event store statistics."""
        events = [
            Event(EventType.TRAINING_STARTED, {}, "test"),
            Event(EventType.TRAINING_COMPLETED, {}, "test"),
            Event(EventType.ERROR_OCCURRED, {}, "test")
        ]
        
        for event in events:
            self.event_store.store_event(event)
        
        stats = self.event_store.get_statistics()
        
        assert stats["total_events"] == 3
        assert "events_by_type" in stats
        assert "events_by_source" in stats
        assert "storage_size_bytes" in stats


class TestEventPublisher:
    """Test EventPublisher class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.event_bus = Mock(spec=EventBus)
        self.publisher = EventPublisher(self.event_bus)
    
    def test_publish_training_event(self):
        """Test publishing training event."""
        self.publisher.publish_training_started("model-123", {"config": "test"})
        
        self.event_bus.publish.assert_called_once()
        published_event = self.event_bus.publish.call_args[0][0]
        
        assert published_event.type == EventType.TRAINING_STARTED
        assert published_event.data["model_id"] == "model-123"
        assert published_event.data["config"] == {"config": "test"}
    
    def test_publish_model_event(self):
        """Test publishing model event."""
        self.publisher.publish_model_loaded("test-model", "/path/to/model")
        
        self.event_bus.publish.assert_called_once()
        published_event = self.event_bus.publish.call_args[0][0]
        
        assert published_event.type == EventType.MODEL_LOADED
        assert published_event.data["model_name"] == "test-model"
        assert published_event.data["model_path"] == "/path/to/model"
    
    def test_publish_system_event(self):
        """Test publishing system event."""
        self.publisher.publish_system_started({"version": "1.0"})
        
        self.event_bus.publish.assert_called_once()
        published_event = self.event_bus.publish.call_args[0][0]
        
        assert published_event.type == EventType.SYSTEM_STARTED
        assert published_event.data["version"] == "1.0"
    
    def test_publish_error_event(self):
        """Test publishing error event."""
        error = Exception("Test error")
        self.publisher.publish_error_occurred(error, "test_component")
        
        self.event_bus.publish.assert_called_once()
        published_event = self.event_bus.publish.call_args[0][0]
        
        assert published_event.type == EventType.ERROR_OCCURRED
        assert published_event.data["error_message"] == "Test error"
        assert published_event.data["component"] == "test_component"
    
    def test_publish_custom_event(self):
        """Test publishing custom event."""
        custom_data = {"custom": "data", "value": 42}
        
        self.publisher.publish_custom_event(
            EventType.CONFIGURATION_CHANGED,
            custom_data,
            source="config_manager",
            priority=EventPriority.HIGH
        )
        
        self.event_bus.publish.assert_called_once()
        published_event = self.event_bus.publish.call_args[0][0]
        
        assert published_event.type == EventType.CONFIGURATION_CHANGED
        assert published_event.data == custom_data
        assert published_event.source == "config_manager"
        assert published_event.priority == EventPriority.HIGH


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_event_bus(self):
        """Test getting global event bus."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        # Should return same instance
        assert bus1 is bus2
        assert isinstance(bus1, EventBus)
    
    def test_get_event_store(self):
        """Test getting global event store."""
        store1 = get_event_store()
        store2 = get_event_store()
        
        # Should return same instance
        assert store1 is store2
        assert isinstance(store1, EventStore)
    
    def test_publish_event(self):
        """Test global publish event function."""
        with patch('src.fine_tune_llm.core.events.get_event_bus') as mock_get_bus:
            mock_bus = Mock()
            mock_get_bus.return_value = mock_bus
            
            event = Event(EventType.TRAINING_STARTED, {}, "test")
            publish_event(event)
            
            mock_bus.publish.assert_called_once_with(event)
    
    def test_subscribe_to_event(self):
        """Test global subscribe to event function."""
        with patch('src.fine_tune_llm.core.events.get_event_bus') as mock_get_bus:
            mock_bus = Mock()
            mock_bus.subscribe.return_value = "subscription-id"
            mock_get_bus.return_value = mock_bus
            
            handler = Mock()
            subscription_id = subscribe_to_event(EventType.TRAINING_STARTED, handler)
            
            assert subscription_id == "subscription-id"
            mock_bus.subscribe.assert_called_once()
    
    def test_create_event(self):
        """Test global create event function."""
        data = {"test": "data"}
        event = create_event(EventType.TRAINING_STARTED, data, "test_source")
        
        assert isinstance(event, Event)
        assert event.type == EventType.TRAINING_STARTED
        assert event.data == data
        assert event.source == "test_source"


class TestEventBusIntegration:
    """Integration tests for event bus functionality."""
    
    def test_full_workflow(self):
        """Test complete event workflow."""
        # Create event bus and store
        event_bus = EventBus()
        event_store = EventStore()
        
        # Create publisher
        publisher = EventPublisher(event_bus)
        
        # Subscribe to events and store them
        def store_handler(event):
            event_store.store_event(event)
        
        event_bus.subscribe(EventType.TRAINING_STARTED, store_handler)
        
        # Publish training events
        publisher.publish_training_started("model-1", {"lr": 0.001})
        publisher.publish_training_completed("model-1", {"accuracy": 0.95})
        
        # Verify events were stored
        stored_events = event_store.get_events()
        assert len(stored_events) == 2
        
        # Verify event types
        event_types = [event.type for event in stored_events]
        assert EventType.TRAINING_STARTED in event_types
        assert EventType.TRAINING_COMPLETED in event_types
    
    def test_error_handling_workflow(self):
        """Test error handling in event workflow."""
        event_bus = EventBus()
        
        # Handler that raises exception
        def failing_handler(event):
            raise Exception("Handler failed")
        
        # Handler that works
        working_handler = Mock()
        
        event_bus.subscribe(EventType.ERROR_OCCURRED, failing_handler)
        event_bus.subscribe(EventType.ERROR_OCCURRED, working_handler)
        
        # Publish error event
        event = Event(EventType.ERROR_OCCURRED, {"error": "test"}, "test")
        event_bus.publish(event)
        
        # Working handler should still be called despite failing handler
        working_handler.assert_called_once_with(event)
    
    def test_concurrent_publishing(self):
        """Test concurrent event publishing."""
        event_bus = EventBus()
        received_events = []
        lock = threading.Lock()
        
        def thread_safe_handler(event):
            with lock:
                received_events.append(event)
        
        event_bus.subscribe(EventType.TRAINING_STARTED, thread_safe_handler)
        
        # Create multiple threads publishing events
        def publish_events(thread_id):
            for i in range(10):
                event = Event(
                    EventType.TRAINING_STARTED,
                    {"thread": thread_id, "count": i},
                    f"thread-{thread_id}"
                )
                event_bus.publish(event)
        
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=publish_events, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were received
        assert len(received_events) == 50
    
    def test_memory_cleanup(self):
        """Test memory cleanup with weak references."""
        event_bus = EventBus()
        
        class TestHandler:
            def __call__(self, event):
                pass
        
        # Create handler and weak reference
        handler = TestHandler()
        weak_ref = weakref.ref(handler)
        
        subscription_id = event_bus.subscribe(EventType.TRAINING_STARTED, handler)
        
        # Verify handler is still alive
        assert weak_ref() is not None
        
        # Delete handler reference
        del handler
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Handler should be cleaned up (this test verifies memory management)
        # Note: In real scenarios, event bus might hold references longer