"""
Time-related utility functions.

This module provides functions for time formatting, duration calculations,
timestamps, and timing operations.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Union, Optional, Dict, Any
import re
from contextlib import contextmanager


def get_timestamp(include_microseconds: bool = False, utc: bool = True) -> str:
    """
    Get current timestamp as ISO format string.
    
    Args:
        include_microseconds: Include microseconds in timestamp
        utc: Use UTC timezone (True) or local timezone (False)
        
    Returns:
        ISO format timestamp string
    """
    if utc:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now()
    
    if include_microseconds:
        return dt.isoformat()
    else:
        return dt.replace(microsecond=0).isoformat()


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Number of decimal places for seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 0:
        return "0s"
    
    # Handle very small durations
    if seconds < 1:
        if seconds < 0.001:
            return f"{seconds * 1000000:.{precision}f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.{precision}f}ms"
    
    # Convert to timedelta for easy formatting
    td = timedelta(seconds=seconds)
    
    # Extract components
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    microseconds = td.microseconds
    
    # Format based on magnitude
    parts = []
    
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    # Include fractional seconds if less than a minute total
    if days == 0 and hours == 0 and minutes == 0:
        total_seconds = secs + microseconds / 1000000
        parts.append(f"{total_seconds:.{precision}f}s")
    elif secs > 0 or not parts:  # Include seconds if present or if it's the only unit
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to seconds.
    
    Supports formats like:
    - "1h 30m 45s"
    - "90m"
    - "3600s"
    - "1.5h"
    - "30.5"  (assumes seconds)
    
    Args:
        duration_str: Duration string to parse
        
    Returns:
        Duration in seconds
        
    Raises:
        ValueError: If duration string format is invalid
    """
    duration_str = duration_str.strip().lower()
    
    # Try to parse as plain number (seconds)
    try:
        return float(duration_str)
    except ValueError:
        pass
    
    # Parse with units
    total_seconds = 0.0
    
    # Regular expression to match number + unit pairs
    pattern = r'(\d+(?:\.\d+)?)\s*([dhms]|ms|μs|us)'
    matches = re.findall(pattern, duration_str)
    
    if not matches:
        raise ValueError(f"Invalid duration format: {duration_str}")
    
    unit_multipliers = {
        'μs': 1e-6,
        'us': 1e-6,  # Alternative microsecond notation
        'ms': 1e-3,
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400
    }
    
    for value_str, unit in matches:
        value = float(value_str)
        
        if unit not in unit_multipliers:
            raise ValueError(f"Unknown time unit: {unit}")
        
        total_seconds += value * unit_multipliers[unit]
    
    return total_seconds


@contextmanager
def timer():
    """
    Context manager for timing code execution.
    
    Usage:
        with timer() as t:
            # some code
            pass
        print(f"Execution took {t.elapsed:.2f} seconds")
        
    Yields:
        Timer object with elapsed time
    """
    class Timer:
        def __init__(self):
            self.start_time = time.perf_counter()
            self.elapsed = 0
        
        def stop(self):
            self.elapsed = time.perf_counter() - self.start_time
    
    timer_obj = Timer()
    try:
        yield timer_obj
    finally:
        timer_obj.stop()


def time_function(func, *args, **kwargs) -> tuple:
    """
    Time a function execution and return result with timing.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


def benchmark_function(func, iterations: int = 1000, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function over multiple iterations.
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark statistics
    """
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times.sort()
    total_time = sum(times)
    
    return {
        'iterations': iterations,
        'total_time': total_time,
        'average_time': total_time / iterations,
        'min_time': times[0],
        'max_time': times[-1],
        'median_time': times[len(times) // 2],
        'p95_time': times[int(0.95 * len(times))],
        'p99_time': times[int(0.99 * len(times))]
    }


def sleep_until(target_time: datetime) -> None:
    """
    Sleep until a specific target time.
    
    Args:
        target_time: Target datetime to sleep until
    """
    now = datetime.now(target_time.tzinfo) if target_time.tzinfo else datetime.now()
    sleep_duration = (target_time - now).total_seconds()
    
    if sleep_duration > 0:
        time.sleep(sleep_duration)


def format_timestamp(timestamp: Union[float, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp to string.
    
    Args:
        timestamp: Unix timestamp (float) or datetime object
        format_str: Format string for datetime formatting
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return dt.strftime(format_str)


def get_unix_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()


def datetime_to_unix(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def unix_to_datetime(timestamp: float, utc: bool = True) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp
        utc: Return UTC datetime if True, local if False
        
    Returns:
        Datetime object
    """
    if utc:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        return datetime.fromtimestamp(timestamp)


def time_ago(timestamp: Union[float, datetime]) -> str:
    """
    Get human-readable "time ago" string.
    
    Args:
        timestamp: Unix timestamp or datetime object
        
    Returns:
        Human-readable time ago string (e.g., "2 hours ago")
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    now = datetime.now()
    diff = now - dt
    
    if diff.total_seconds() < 0:
        return "in the future"
    
    seconds = int(diff.total_seconds())
    
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''} ago"
    
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    
    days = hours // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    
    years = months // 12
    return f"{years} year{'s' if years != 1 else ''} ago"


def is_recent(timestamp: Union[float, datetime], threshold_seconds: float = 300) -> bool:
    """
    Check if timestamp is recent (within threshold).
    
    Args:
        timestamp: Unix timestamp or datetime object
        threshold_seconds: Threshold in seconds (default: 5 minutes)
        
    Returns:
        True if timestamp is within threshold of current time
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    now = datetime.now()
    diff = abs((now - dt).total_seconds())
    
    return diff <= threshold_seconds


class PerformanceTimer:
    """Performance timer for measuring execution time with context management."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize performance timer.
        
        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def start(self) -> 'PerformanceTimer':
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self) -> 'PerformanceTimer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
        if self.name:
            print(f"{self.name}: {format_duration(self.elapsed)}")
    
    def __str__(self) -> str:
        """String representation."""
        if self.elapsed is None:
            return f"Timer({self.name or 'unnamed'}): not finished"
        else:
            return f"Timer({self.name or 'unnamed'}): {format_duration(self.elapsed)}"