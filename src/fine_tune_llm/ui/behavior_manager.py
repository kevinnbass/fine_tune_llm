"""
Behavior Manager for consistent UI behavior across components.

This module provides centralized behavior management including keyboard shortcuts,
interaction patterns, state management, and consistent user experience patterns.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of UI interactions."""
    CLICK = "click"
    HOVER = "hover"
    FOCUS = "focus"
    KEYDOWN = "keydown"
    SCROLL = "scroll"
    RESIZE = "resize"


class BehaviorEvent(Enum):
    """Standard behavior events."""
    COMPONENT_LOADED = "component_loaded"
    COMPONENT_UNLOADED = "component_unloaded"
    THEME_CHANGED = "theme_changed"
    CONFIG_CHANGED = "config_changed"
    ERROR_OCCURRED = "error_occurred"
    DATA_UPDATED = "data_updated"


@dataclass
class KeyboardShortcut:
    """Keyboard shortcut configuration."""
    keys: str  # e.g., "Ctrl+S", "Alt+F4"
    action: str
    description: str
    scope: str = "global"  # "global", "component", or component name
    enabled: bool = True


@dataclass
class InteractionRule:
    """UI interaction rule."""
    trigger: InteractionType
    target: str  # CSS selector or element type
    action: Callable
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass 
class StateTransition:
    """State transition rule."""
    from_state: str
    to_state: str
    trigger: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None


class BehaviorManager:
    """
    Centralized behavior management system.
    
    Manages consistent UI behavior patterns, keyboard shortcuts,
    interaction rules, and state management across all components.
    """
    
    def __init__(self):
        """Initialize behavior manager."""
        # Keyboard shortcuts
        self.shortcuts: Dict[str, KeyboardShortcut] = {}
        self.shortcut_handlers: Dict[str, Callable] = {}
        
        # Interaction rules
        self.interaction_rules: List[InteractionRule] = []
        
        # State management
        self.component_states: Dict[str, str] = {}
        self.state_transitions: Dict[str, List[StateTransition]] = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Behavior patterns
        self.patterns: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default behaviors
        self._initialize_default_behaviors()
        
        logger.info("Initialized BehaviorManager")
    
    def _initialize_default_behaviors(self):
        """Initialize default UI behavior patterns."""
        # Default keyboard shortcuts
        default_shortcuts = [
            KeyboardShortcut("Ctrl+S", "save", "Save current work"),
            KeyboardShortcut("Ctrl+R", "refresh", "Refresh data"),
            KeyboardShortcut("Ctrl+F", "search", "Open search"),
            KeyboardShortcut("Escape", "close", "Close modal/dialog"),
            KeyboardShortcut("F1", "help", "Show help"),
            KeyboardShortcut("Ctrl+Z", "undo", "Undo last action"),
            KeyboardShortcut("Ctrl+Y", "redo", "Redo last action"),
            KeyboardShortcut("Ctrl+D", "download", "Download data"),
            KeyboardShortcut("Tab", "navigate", "Navigate between elements"),
            KeyboardShortcut("Enter", "confirm", "Confirm action")
        ]
        
        for shortcut in default_shortcuts:
            self.register_shortcut(shortcut)
        
        # Default interaction patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default interaction patterns."""
        patterns = {
            'form_validation': {
                'on_blur_validate': True,
                'show_errors_inline': True,
                'highlight_invalid_fields': True,
                'prevent_submit_if_invalid': True
            },
            'data_loading': {
                'show_loading_spinner': True,
                'disable_interactions_while_loading': True,
                'auto_retry_on_failure': True,
                'show_progress_for_long_operations': True
            },
            'navigation': {
                'smooth_scroll': True,
                'highlight_active_nav': True,
                'breadcrumb_navigation': True,
                'back_button_behavior': 'browser_history'
            },
            'modals': {
                'close_on_escape': True,
                'close_on_backdrop_click': True,
                'focus_trap': True,
                'restore_focus_on_close': True
            },
            'tables': {
                'sortable_headers': True,
                'row_hover_highlight': True,
                'pagination_controls': True,
                'column_resize': True
            },
            'notifications': {
                'auto_dismiss_delay': 5000,
                'stack_multiple': True,
                'position': 'top-right',
                'animation': 'slide-in'
            }
        }
        
        with self._lock:
            self.patterns.update(patterns)
    
    def register_shortcut(self, shortcut: KeyboardShortcut) -> bool:
        """
        Register keyboard shortcut.
        
        Args:
            shortcut: Keyboard shortcut configuration
            
        Returns:
            True if registered successfully
        """
        try:
            with self._lock:
                key = f"{shortcut.scope}:{shortcut.keys}"
                self.shortcuts[key] = shortcut
            
            logger.debug(f"Registered keyboard shortcut: {shortcut.keys} -> {shortcut.action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register shortcut {shortcut.keys}: {e}")
            return False
    
    def unregister_shortcut(self, keys: str, scope: str = "global") -> bool:
        """
        Unregister keyboard shortcut.
        
        Args:
            keys: Keyboard combination
            scope: Shortcut scope
            
        Returns:
            True if unregistered successfully
        """
        try:
            with self._lock:
                key = f"{scope}:{keys}"
                if key in self.shortcuts:
                    del self.shortcuts[key]
                    if key in self.shortcut_handlers:
                        del self.shortcut_handlers[key]
            
            logger.debug(f"Unregistered keyboard shortcut: {keys}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister shortcut {keys}: {e}")
            return False
    
    def bind_shortcut_handler(self, keys: str, handler: Callable, scope: str = "global") -> bool:
        """
        Bind handler to keyboard shortcut.
        
        Args:
            keys: Keyboard combination
            handler: Handler function
            scope: Shortcut scope
            
        Returns:
            True if bound successfully
        """
        try:
            with self._lock:
                key = f"{scope}:{keys}"
                if key in self.shortcuts:
                    self.shortcut_handlers[key] = handler
                    logger.debug(f"Bound handler to shortcut: {keys}")
                    return True
                else:
                    logger.warning(f"Shortcut not registered: {keys}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to bind shortcut handler {keys}: {e}")
            return False
    
    def handle_keyboard_event(self, keys: str, scope: str = "global") -> bool:
        """
        Handle keyboard event.
        
        Args:
            keys: Keyboard combination pressed
            scope: Event scope
            
        Returns:
            True if event was handled
        """
        try:
            key = f"{scope}:{keys}"
            
            with self._lock:
                if key in self.shortcut_handlers and key in self.shortcuts:
                    shortcut = self.shortcuts[key]
                    if shortcut.enabled:
                        handler = self.shortcut_handlers[key]
                        handler()
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling keyboard event {keys}: {e}")
            return False
    
    def get_shortcuts_for_scope(self, scope: str) -> List[KeyboardShortcut]:
        """
        Get all shortcuts for a scope.
        
        Args:
            scope: Scope name
            
        Returns:
            List of shortcuts
        """
        with self._lock:
            return [
                shortcut for key, shortcut in self.shortcuts.items()
                if shortcut.scope == scope and shortcut.enabled
            ]
    
    def add_interaction_rule(self, rule: InteractionRule):
        """
        Add interaction rule.
        
        Args:
            rule: Interaction rule
        """
        with self._lock:
            self.interaction_rules.append(rule)
        
        logger.debug(f"Added interaction rule: {rule.trigger.value} on {rule.target}")
    
    def remove_interaction_rule(self, rule: InteractionRule):
        """
        Remove interaction rule.
        
        Args:
            rule: Interaction rule to remove
        """
        with self._lock:
            if rule in self.interaction_rules:
                self.interaction_rules.remove(rule)
        
        logger.debug(f"Removed interaction rule: {rule.trigger.value} on {rule.target}")
    
    def get_interaction_rules_for_trigger(self, trigger: InteractionType) -> List[InteractionRule]:
        """
        Get interaction rules for trigger type.
        
        Args:
            trigger: Interaction trigger type
            
        Returns:
            List of matching rules
        """
        with self._lock:
            return [rule for rule in self.interaction_rules 
                   if rule.trigger == trigger and rule.enabled]
    
    def set_component_state(self, component: str, state: str):
        """
        Set component state.
        
        Args:
            component: Component name
            state: New state
        """
        with self._lock:
            old_state = self.component_states.get(component)
            self.component_states[component] = state
            
            # Check for state transitions
            if old_state and old_state != state:
                self._handle_state_transition(component, old_state, state)
        
        logger.debug(f"Set {component} state: {old_state} -> {state}")
    
    def get_component_state(self, component: str) -> Optional[str]:
        """
        Get component state.
        
        Args:
            component: Component name
            
        Returns:
            Current state or None
        """
        with self._lock:
            return self.component_states.get(component)
    
    def add_state_transition(self, component: str, transition: StateTransition):
        """
        Add state transition rule.
        
        Args:
            component: Component name
            transition: State transition rule
        """
        with self._lock:
            if component not in self.state_transitions:
                self.state_transitions[component] = []
            self.state_transitions[component].append(transition)
        
        logger.debug(f"Added state transition for {component}: {transition.from_state} -> {transition.to_state}")
    
    def _handle_state_transition(self, component: str, from_state: str, to_state: str):
        """Handle state transition."""
        with self._lock:
            transitions = self.state_transitions.get(component, [])
            
            for transition in transitions:
                if (transition.from_state == from_state and 
                    transition.to_state == to_state):
                    
                    # Check condition if present
                    if transition.condition and not transition.condition():
                        continue
                    
                    # Execute action if present
                    if transition.action:
                        try:
                            transition.action()
                        except Exception as e:
                            logger.error(f"Error executing state transition action: {e}")
    
    def register_event_handler(self, event: Union[str, BehaviorEvent], handler: Callable):
        """
        Register event handler.
        
        Args:
            event: Event name or type
            handler: Handler function
        """
        event_name = event.value if isinstance(event, BehaviorEvent) else event
        
        with self._lock:
            if event_name not in self.event_handlers:
                self.event_handlers[event_name] = []
            self.event_handlers[event_name].append(handler)
        
        logger.debug(f"Registered event handler for: {event_name}")
    
    def unregister_event_handler(self, event: Union[str, BehaviorEvent], handler: Callable):
        """
        Unregister event handler.
        
        Args:
            event: Event name or type
            handler: Handler function
        """
        event_name = event.value if isinstance(event, BehaviorEvent) else event
        
        with self._lock:
            if event_name in self.event_handlers:
                handlers = self.event_handlers[event_name]
                if handler in handlers:
                    handlers.remove(handler)
                    if not handlers:
                        del self.event_handlers[event_name]
        
        logger.debug(f"Unregistered event handler for: {event_name}")
    
    def emit_event(self, event: Union[str, BehaviorEvent], data: Optional[Dict[str, Any]] = None):
        """
        Emit event to registered handlers.
        
        Args:
            event: Event name or type
            data: Event data
        """
        event_name = event.value if isinstance(event, BehaviorEvent) else event
        
        with self._lock:
            handlers = self.event_handlers.get(event_name, [])
        
        for handler in handlers:
            try:
                if data:
                    handler(data)
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
    
    def get_behavior_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """
        Get behavior pattern configuration.
        
        Args:
            pattern_name: Pattern name
            
        Returns:
            Pattern configuration or None
        """
        with self._lock:
            return self.patterns.get(pattern_name, {}).copy()
    
    def set_behavior_pattern(self, pattern_name: str, config: Dict[str, Any]):
        """
        Set behavior pattern configuration.
        
        Args:
            pattern_name: Pattern name
            config: Pattern configuration
        """
        with self._lock:
            self.patterns[pattern_name] = config.copy()
        
        logger.debug(f"Updated behavior pattern: {pattern_name}")
    
    def update_behavior_pattern(self, pattern_name: str, updates: Dict[str, Any]):
        """
        Update behavior pattern configuration.
        
        Args:
            pattern_name: Pattern name
            updates: Configuration updates
        """
        with self._lock:
            if pattern_name in self.patterns:
                self.patterns[pattern_name].update(updates)
            else:
                self.patterns[pattern_name] = updates.copy()
        
        logger.debug(f"Updated behavior pattern: {pattern_name}")
    
    def generate_javascript(self) -> str:
        """
        Generate JavaScript for behavior management.
        
        Returns:
            JavaScript code for client-side behavior
        """
        shortcuts_js = self._generate_shortcuts_javascript()
        patterns_js = self._generate_patterns_javascript()
        interactions_js = self._generate_interactions_javascript()
        
        return f"""
// Behavior Manager JavaScript
(function() {{
    'use strict';
    
    // Global behavior manager
    window.BehaviorManager = {{
        shortcuts: {{}},
        patterns: {json.dumps(self.patterns, indent=2)},
        
        init: function() {{
            this.initShortcuts();
            this.initPatterns();
            this.initInteractions();
        }},
        
        {shortcuts_js}
        
        {patterns_js}
        
        {interactions_js}
        
        // Utility functions
        debounce: function(func, wait) {{
            let timeout;
            return function executedFunction(...args) {{
                const later = () => {{
                    clearTimeout(timeout);
                    func(...args);
                }};
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            }};
        }},
        
        throttle: function(func, limit) {{
            let inThrottle;
            return function() {{
                const args = arguments;
                const context = this;
                if (!inThrottle) {{
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }}
            }}
        }}
    }};
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', () => {{
            window.BehaviorManager.init();
        }});
    }} else {{
        window.BehaviorManager.init();
    }}
}})();
"""
    
    def _generate_shortcuts_javascript(self) -> str:
        """Generate JavaScript for keyboard shortcuts."""
        shortcuts_data = []
        with self._lock:
            for key, shortcut in self.shortcuts.items():
                if shortcut.enabled:
                    shortcuts_data.append({
                        'keys': shortcut.keys,
                        'action': shortcut.action,
                        'scope': shortcut.scope
                    })
        
        return f"""
        initShortcuts: function() {{
            const shortcuts = {json.dumps(shortcuts_data, indent=4)};
            
            document.addEventListener('keydown', (e) => {{
                const key = this.getKeyCombo(e);
                const scope = this.getCurrentScope();
                
                shortcuts.forEach(shortcut => {{
                    if (shortcut.keys === key && (shortcut.scope === 'global' || shortcut.scope === scope)) {{
                        e.preventDefault();
                        this.handleShortcut(shortcut.action, e);
                    }}
                }});
            }});
        }},
        
        getKeyCombo: function(e) {{
            let combo = '';
            if (e.ctrlKey) combo += 'Ctrl+';
            if (e.altKey) combo += 'Alt+';
            if (e.shiftKey) combo += 'Shift+';
            if (e.metaKey) combo += 'Meta+';
            combo += e.key;
            return combo;
        }},
        
        getCurrentScope: function() {{
            const activeElement = document.activeElement;
            return activeElement.dataset.scope || 'global';
        }},
        
        handleShortcut: function(action, event) {{
            const customEvent = new CustomEvent('shortcut:' + action, {{
                detail: {{ originalEvent: event }}
            }});
            document.dispatchEvent(customEvent);
        }},
"""
    
    def _generate_patterns_javascript(self) -> str:
        """Generate JavaScript for behavior patterns."""
        return """
        initPatterns: function() {
            this.initFormValidation();
            this.initDataLoading();
            this.initNavigation();
            this.initModals();
            this.initTables();
            this.initNotifications();
        },
        
        initFormValidation: function() {
            if (!this.patterns.form_validation) return;
            
            const pattern = this.patterns.form_validation;
            
            if (pattern.on_blur_validate) {
                document.addEventListener('blur', (e) => {
                    if (e.target.matches('input, select, textarea')) {
                        this.validateField(e.target);
                    }
                }, true);
            }
        },
        
        validateField: function(field) {
            // Basic validation logic
            const isValid = field.checkValidity();
            
            if (this.patterns.form_validation.highlight_invalid_fields) {
                field.classList.toggle('invalid', !isValid);
            }
            
            if (this.patterns.form_validation.show_errors_inline) {
                this.showFieldError(field, field.validationMessage);
            }
        },
        
        showFieldError: function(field, message) {
            let errorElement = field.parentNode.querySelector('.field-error');
            
            if (message && !field.validity.valid) {
                if (!errorElement) {
                    errorElement = document.createElement('div');
                    errorElement.className = 'field-error';
                    field.parentNode.appendChild(errorElement);
                }
                errorElement.textContent = message;
                errorElement.style.display = 'block';
            } else if (errorElement) {
                errorElement.style.display = 'none';
            }
        },
        
        initDataLoading: function() {
            const pattern = this.patterns.data_loading;
            
            // Intercept fetch requests to show loading states
            const originalFetch = window.fetch;
            window.fetch = (...args) => {
                if (pattern.show_loading_spinner) {
                    this.showLoading();
                }
                
                return originalFetch(...args)
                    .then(response => {
                        this.hideLoading();
                        return response;
                    })
                    .catch(error => {
                        this.hideLoading();
                        if (pattern.auto_retry_on_failure) {
                            // Implement retry logic
                        }
                        throw error;
                    });
            };
        },
        
        showLoading: function() {
            let loader = document.querySelector('.global-loader');
            if (!loader) {
                loader = document.createElement('div');
                loader.className = 'global-loader';
                loader.innerHTML = '<div class="spinner"></div>';
                document.body.appendChild(loader);
            }
            loader.style.display = 'flex';
        },
        
        hideLoading: function() {
            const loader = document.querySelector('.global-loader');
            if (loader) {
                loader.style.display = 'none';
            }
        },
        
        initNavigation: function() {
            const pattern = this.patterns.navigation;
            
            if (pattern.smooth_scroll) {
                document.documentElement.style.scrollBehavior = 'smooth';
            }
            
            if (pattern.highlight_active_nav) {
                this.updateActiveNav();
                window.addEventListener('popstate', () => this.updateActiveNav());
            }
        },
        
        updateActiveNav: function() {
            const currentPath = window.location.pathname;
            document.querySelectorAll('nav a').forEach(link => {
                link.classList.toggle('active', link.pathname === currentPath);
            });
        },
        
        initModals: function() {
            const pattern = this.patterns.modals;
            
            document.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (!modal) return;
                
                if (pattern.close_on_backdrop_click && e.target === modal) {
                    this.closeModal(modal);
                }
            });
            
            if (pattern.close_on_escape) {
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape') {
                        const modal = document.querySelector('.modal.active');
                        if (modal) {
                            this.closeModal(modal);
                        }
                    }
                });
            }
        },
        
        closeModal: function(modal) {
            modal.classList.remove('active');
            
            if (this.patterns.modals.restore_focus_on_close) {
                const trigger = modal.dataset.trigger;
                if (trigger) {
                    document.querySelector(trigger)?.focus();
                }
            }
        },
        
        initTables: function() {
            const pattern = this.patterns.tables;
            
            if (pattern.sortable_headers) {
                document.addEventListener('click', (e) => {
                    if (e.target.matches('th[data-sortable]')) {
                        this.sortTable(e.target);
                    }
                });
            }
            
            if (pattern.row_hover_highlight) {
                document.addEventListener('mouseover', (e) => {
                    if (e.target.closest('tbody tr')) {
                        e.target.closest('tr').classList.add('hover');
                    }
                });
                
                document.addEventListener('mouseout', (e) => {
                    if (e.target.closest('tbody tr')) {
                        e.target.closest('tr').classList.remove('hover');
                    }
                });
            }
        },
        
        sortTable: function(header) {
            // Basic table sorting implementation
            const table = header.closest('table');
            const columnIndex = Array.from(header.parentNode.children).indexOf(header);
            const direction = header.dataset.sortDirection === 'asc' ? 'desc' : 'asc';
            
            header.dataset.sortDirection = direction;
            
            // Sort rows
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            rows.sort((a, b) => {
                const aValue = a.cells[columnIndex].textContent.trim();
                const bValue = b.cells[columnIndex].textContent.trim();
                
                const result = aValue.localeCompare(bValue, undefined, { numeric: true });
                return direction === 'asc' ? result : -result;
            });
            
            rows.forEach(row => tbody.appendChild(row));
        },
        
        initNotifications: function() {
            const pattern = this.patterns.notifications;
            
            this.notificationContainer = document.createElement('div');
            this.notificationContainer.className = 'notification-container';
            this.notificationContainer.style.position = 'fixed';
            
            // Set position
            const [vertical, horizontal] = pattern.position.split('-');
            this.notificationContainer.style[vertical] = '20px';
            this.notificationContainer.style[horizontal] = '20px';
            
            document.body.appendChild(this.notificationContainer);
        },
        
        showNotification: function(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            
            this.notificationContainer.appendChild(notification);
            
            // Auto-dismiss
            const delay = this.patterns.notifications.auto_dismiss_delay;
            if (delay > 0) {
                setTimeout(() => {
                    this.dismissNotification(notification);
                }, delay);
            }
            
            return notification;
        },
        
        dismissNotification: function(notification) {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        },
"""
    
    def _generate_interactions_javascript(self) -> str:
        """Generate JavaScript for interaction rules."""
        return """
        initInteractions: function() {
            // Custom interaction rules would be added here
            this.initCustomInteractions();
        },
        
        initCustomInteractions: function() {
            // Placeholder for custom interaction rules
            // These would be generated based on registered interaction rules
        }
"""
    
    def export_config(self, export_path: Path) -> bool:
        """
        Export behavior configuration to file.
        
        Args:
            export_path: Export file path
            
        Returns:
            True if exported successfully
        """
        try:
            config = {
                'shortcuts': {
                    key: {
                        'keys': shortcut.keys,
                        'action': shortcut.action,
                        'description': shortcut.description,
                        'scope': shortcut.scope,
                        'enabled': shortcut.enabled
                    }
                    for key, shortcut in self.shortcuts.items()
                },
                'patterns': self.patterns.copy(),
                'component_states': self.component_states.copy()
            }
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Exported behavior config to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export behavior config: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """
        Import behavior configuration from file.
        
        Args:
            import_path: Import file path
            
        Returns:
            True if imported successfully
        """
        try:
            with open(import_path, 'r') as f:
                config = json.load(f)
            
            # Import shortcuts
            if 'shortcuts' in config:
                for key, shortcut_data in config['shortcuts'].items():
                    shortcut = KeyboardShortcut(**shortcut_data)
                    self.shortcuts[key] = shortcut
            
            # Import patterns
            if 'patterns' in config:
                self.patterns.update(config['patterns'])
            
            # Import component states
            if 'component_states' in config:
                self.component_states.update(config['component_states'])
            
            logger.info(f"Imported behavior config from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import behavior config: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get behavior manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                'shortcuts_count': len(self.shortcuts),
                'active_shortcuts': len([s for s in self.shortcuts.values() if s.enabled]),
                'interaction_rules': len(self.interaction_rules),
                'patterns_count': len(self.patterns),
                'component_states': len(self.component_states),
                'event_handlers': sum(len(handlers) for handlers in self.event_handlers.values()),
                'state_transitions': sum(len(transitions) for transitions in self.state_transitions.values())
            }
    
    def close(self):
        """Clean up behavior manager."""
        try:
            with self._lock:
                self.shortcuts.clear()
                self.shortcut_handlers.clear()
                self.interaction_rules.clear()
                self.event_handlers.clear()
                self.component_states.clear()
                self.state_transitions.clear()
            
            logger.info("Behavior manager closed")
            
        except Exception as e:
            logger.error(f"Error closing behavior manager: {e}")


# Singleton instance
_behavior_manager = None

def get_behavior_manager() -> BehaviorManager:
    """Get global behavior manager instance."""
    global _behavior_manager
    if _behavior_manager is None:
        _behavior_manager = BehaviorManager()
    return _behavior_manager