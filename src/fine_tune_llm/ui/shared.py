"""
Shared UI utilities and common components.

This module provides reusable UI utilities, widgets, and helper functions
that can be shared across all UI components.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import base64

logger = logging.getLogger(__name__)


class SharedWidgets:
    """Collection of reusable UI widgets."""
    
    @staticmethod
    def metric_card(title: str, 
                   value: Union[str, float, int], 
                   delta: Optional[Union[str, float, int]] = None,
                   delta_color: str = "normal",
                   help_text: Optional[str] = None) -> str:
        """
        Create a metric card widget.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Change from previous value
            delta_color: Color for delta (normal, inverse)
            help_text: Help tooltip text
            
        Returns:
            HTML string for metric card
        """
        delta_html = ""
        if delta is not None:
            delta_class = "delta-positive" if (isinstance(delta, (int, float)) and delta >= 0) else "delta-negative"
            if delta_color == "inverse":
                delta_class = "delta-negative" if delta >= 0 else "delta-positive"
            
            delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
        
        help_html = ""
        if help_text:
            help_html = f'<div class="metric-help" title="{help_text}">?</div>'
        
        return f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-title">{title}</span>
                {help_html}
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def status_badge(status: str, 
                    color: Optional[str] = None,
                    pulse: bool = False) -> str:
        """
        Create a status badge widget.
        
        Args:
            status: Status text
            color: Badge color (success, warning, error, info)
            pulse: Whether to add pulsing animation
            
        Returns:
            HTML string for status badge
        """
        if color is None:
            color_map = {
                'running': 'success',
                'completed': 'success',
                'failed': 'error',
                'warning': 'warning',
                'pending': 'info',
                'stopped': 'warning'
            }
            color = color_map.get(status.lower(), 'info')
        
        pulse_class = " badge-pulse" if pulse else ""
        
        return f"""
        <span class="status-badge badge-{color}{pulse_class}">
            {status}
        </span>
        """
    
    @staticmethod
    def progress_bar(value: float, 
                    max_value: float = 100.0,
                    label: Optional[str] = None,
                    color: str = "primary") -> str:
        """
        Create a progress bar widget.
        
        Args:
            value: Current progress value
            max_value: Maximum value
            label: Progress label
            color: Progress bar color
            
        Returns:
            HTML string for progress bar
        """
        percentage = min(100, (value / max_value) * 100)
        
        label_html = f'<div class="progress-label">{label}</div>' if label else ""
        
        return f"""
        <div class="progress-container">
            {label_html}
            <div class="progress-bar">
                <div class="progress-fill progress-{color}" style="width: {percentage}%"></div>
            </div>
            <div class="progress-text">{value:.1f} / {max_value}</div>
        </div>
        """
    
    @staticmethod
    def data_table(data: List[Dict[str, Any]], 
                  columns: Optional[List[str]] = None,
                  sortable: bool = True,
                  searchable: bool = True,
                  max_height: Optional[str] = "400px") -> str:
        """
        Create a data table widget.
        
        Args:
            data: List of row dictionaries
            columns: Column names to display
            sortable: Whether columns are sortable
            searchable: Whether table is searchable
            max_height: Maximum table height
            
        Returns:
            HTML string for data table
        """
        if not data:
            return '<div class="table-empty">No data available</div>'
        
        if columns is None:
            columns = list(data[0].keys())
        
        # Generate table header
        sortable_class = " sortable" if sortable else ""
        header_html = "<tr>"
        for col in columns:
            header_html += f'<th class="table-header{sortable_class}">{col}</th>'
        header_html += "</tr>"
        
        # Generate table rows
        rows_html = ""
        for row in data:
            rows_html += "<tr>"
            for col in columns:
                value = row.get(col, "")
                # Format value based on type
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                elif isinstance(value, datetime):
                    formatted_value = value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_value = str(value)
                rows_html += f"<td>{formatted_value}</td>"
            rows_html += "</tr>"
        
        # Search box
        search_html = ""
        if searchable:
            search_html = '''
            <div class="table-search">
                <input type="text" placeholder="Search..." class="search-input">
            </div>
            '''
        
        style = f'max-height: {max_height}; overflow-y: auto;' if max_height else ""
        
        return f"""
        <div class="table-container">
            {search_html}
            <div class="table-wrapper" style="{style}">
                <table class="data-table">
                    <thead>{header_html}</thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
        """
    
    @staticmethod
    def alert_box(message: str, 
                 alert_type: str = "info",
                 dismissible: bool = True,
                 title: Optional[str] = None) -> str:
        """
        Create an alert box widget.
        
        Args:
            message: Alert message
            alert_type: Alert type (success, info, warning, error)
            dismissible: Whether alert can be dismissed
            title: Optional alert title
            
        Returns:
            HTML string for alert box
        """
        icons = {
            'success': '✓',
            'info': 'ℹ',
            'warning': '⚠',
            'error': '✕'
        }
        
        icon = icons.get(alert_type, 'ℹ')
        dismiss_html = '<button class="alert-dismiss">×</button>' if dismissible else ""
        title_html = f'<div class="alert-title">{title}</div>' if title else ""
        
        return f"""
        <div class="alert alert-{alert_type}">
            <div class="alert-icon">{icon}</div>
            <div class="alert-content">
                {title_html}
                <div class="alert-message">{message}</div>
            </div>
            {dismiss_html}
        </div>
        """


class UIHelpers:
    """Collection of UI helper functions."""
    
    @staticmethod
    def format_bytes(size: int) -> str:
        """Format byte size as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"
    
    @staticmethod
    def format_number(number: Union[int, float], 
                     precision: int = 2,
                     use_si: bool = True) -> str:
        """Format number with appropriate scale."""
        if use_si and abs(number) >= 1000:
            for threshold, suffix in [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]:
                if abs(number) >= threshold:
                    return f"{number/threshold:.{precision}f}{suffix}"
        
        if isinstance(number, int):
            return f"{number:,}"
        else:
            return f"{number:,.{precision}f}"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def get_color_scale(values: List[float], 
                       color_map: str = "viridis") -> List[str]:
        """Generate color scale for values."""
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        
        # Predefined color maps
        color_maps = {
            'viridis': ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
            'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            'blues': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd'],
            'reds': ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26'],
        }
        
        colors = color_maps.get(color_map, color_maps['viridis'])
        
        # Map values to colors
        result_colors = []
        for value in values:
            if max_val == min_val:
                normalized = 0.5
            else:
                normalized = (value - min_val) / (max_val - min_val)
            
            # Get color from scale
            color_index = int(normalized * (len(colors) - 1))
            result_colors.append(colors[color_index])
        
        return result_colors
    
    @staticmethod
    def create_download_link(data: Union[str, bytes, Dict], 
                           filename: str,
                           mime_type: str = "text/plain") -> str:
        """Create download link for data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, indent=2)
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Create base64 encoded data URL
        b64_data = base64.b64encode(data_bytes).decode()
        href = f"data:{mime_type};base64,{b64_data}"
        
        return f'''
        <a download="{filename}" 
           href="{href}" 
           class="download-link">
            📥 Download {filename}
        </a>
        '''
    
    @staticmethod
    def create_tabs(tabs: Dict[str, str], active_tab: str = None) -> str:
        """Create tabbed interface."""
        if not tabs:
            return ""
        
        if active_tab is None:
            active_tab = list(tabs.keys())[0]
        
        # Tab headers
        tab_headers = []
        for tab_id, tab_title in tabs.items():
            active_class = " active" if tab_id == active_tab else ""
            tab_headers.append(f'''
                <button class="tab-button{active_class}" onclick="showTab('{tab_id}')">
                    {tab_title}
                </button>
            ''')
        
        # Tab content
        tab_contents = []
        for tab_id in tabs.keys():
            active_class = " active" if tab_id == active_tab else ""
            tab_contents.append(f'''
                <div id="tab-{tab_id}" class="tab-content{active_class}">
                    <!-- Tab content will be populated by component -->
                </div>
            ''')
        
        return f'''
        <div class="tabs-container">
            <div class="tab-headers">
                {"".join(tab_headers)}
            </div>
            <div class="tab-body">
                {"".join(tab_contents)}
            </div>
        </div>
        '''


class SharedStyles:
    """Shared CSS styles for UI components."""
    
    @staticmethod
    def get_base_styles() -> str:
        """Get base CSS styles for all components."""
        return """
        /* Base styles */
        .metric-card {
            background: var(--secondary-color);
            border: 1px solid var(--primary-color);
            border-radius: var(--border-radius);
            padding: var(--spacing);
            margin-bottom: calc(var(--spacing) * 0.5);
        }
        
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .metric-title {
            font-weight: bold;
            color: var(--text-color);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-delta {
            font-size: 0.9em;
            margin-top: 0.25rem;
        }
        
        .delta-positive {
            color: #28a745;
        }
        
        .delta-negative {
            color: #dc3545;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: calc(var(--border-radius) * 0.5);
            font-size: 0.875em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .badge-success {
            background-color: #d4edda;
            color: #155724;
        }
        
        .badge-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .badge-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .badge-info {
            background-color: #d1ecf1;
            color: #0c5460;
        }
        
        .badge-pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .progress-container {
            margin-bottom: var(--spacing);
        }
        
        .progress-label {
            margin-bottom: 0.25rem;
            font-weight: bold;
        }
        
        .progress-bar {
            background-color: var(--secondary-color);
            border-radius: var(--border-radius);
            height: 1.5rem;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .progress-primary {
            background-color: var(--primary-color);
        }
        
        .progress-text {
            text-align: center;
            font-size: 0.875em;
            margin-top: 0.25rem;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--background-color);
        }
        
        .data-table th,
        .data-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--secondary-color);
        }
        
        .data-table th {
            background: var(--secondary-color);
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        
        .data-table tr:hover {
            background-color: var(--secondary-color);
        }
        
        .alert {
            display: flex;
            align-items: flex-start;
            padding: var(--spacing);
            border-radius: var(--border-radius);
            margin-bottom: var(--spacing);
        }
        
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .alert-info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        .alert-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .alert-icon {
            margin-right: 0.75rem;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .alert-content {
            flex: 1;
        }
        
        .alert-title {
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        
        .tabs-container {
            border: 1px solid var(--secondary-color);
            border-radius: var(--border-radius);
        }
        
        .tab-headers {
            display: flex;
            border-bottom: 1px solid var(--secondary-color);
        }
        
        .tab-button {
            padding: 0.75rem 1rem;
            background: none;
            border: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab-button:hover {
            background-color: var(--secondary-color);
        }
        
        .tab-button.active {
            border-bottom-color: var(--primary-color);
            background-color: var(--secondary-color);
        }
        
        .tab-content {
            display: none;
            padding: var(--spacing);
        }
        
        .tab-content.active {
            display: block;
        }
        """
    
    @staticmethod
    def get_javascript() -> str:
        """Get shared JavaScript functions."""
        return """
        function showTab(tabId) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab content
            const selectedContent = document.getElementById('tab-' + tabId);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }
            
            // Mark button as active
            const selectedButton = event.target;
            selectedButton.classList.add('active');
        }
        
        function dismissAlert(element) {
            element.style.display = 'none';
        }
        
        // Add dismiss functionality to alerts
        document.addEventListener('DOMContentLoaded', function() {
            const dismissButtons = document.querySelectorAll('.alert-dismiss');
            dismissButtons.forEach(button => {
                button.addEventListener('click', function() {
                    dismissAlert(this.parentElement);
                });
            });
        });
        """