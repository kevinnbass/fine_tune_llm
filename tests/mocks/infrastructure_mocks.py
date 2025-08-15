"""
Mock implementations for infrastructure components.

This module provides comprehensive mocking for GPU management, monitoring,
notifications, secrets management, and other infrastructure services.
"""

import random
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, MagicMock
from pathlib import Path


class MockGPUManager:
    """Mock GPU manager for testing GPU operations."""
    
    def __init__(self, num_gpus: int = 2):
        self.num_gpus = num_gpus
        self.gpus = self._initialize_gpus()
        self.memory_allocations = {}
        self._monitoring = False
        self._monitor_thread = None
        
    def _initialize_gpus(self):
        """Initialize mock GPU information."""
        gpus = {}
        for i in range(self.num_gpus):
            gpus[i] = {
                "id": i,
                "name": f"Mock GPU {i}",
                "memory_total": 8192,  # 8GB in MB
                "memory_used": random.randint(1000, 3000),
                "memory_free": 0,  # Will be calculated
                "utilization": random.randint(10, 50),
                "temperature": random.randint(40, 70),
                "power_usage": random.randint(50, 200),
                "processes": [],
                "available": True
            }
            gpus[i]["memory_free"] = gpus[i]["memory_total"] - gpus[i]["memory_used"]
        
        return gpus
    
    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        return len([gpu for gpu in self.gpus.values() if gpu["available"]])
    
    def get_gpu_info(self, gpu_id: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Get GPU information."""
        if gpu_id is not None:
            if gpu_id not in self.gpus:
                raise ValueError(f"GPU {gpu_id} not found")
            return self.gpus[gpu_id].copy()
        else:
            return [gpu.copy() for gpu in self.gpus.values()]
    
    def allocate_memory(self, gpu_id: int, size_mb: int, process_name: str = "test_process"):
        """Mock GPU memory allocation."""
        if gpu_id not in self.gpus:
            raise ValueError(f"GPU {gpu_id} not found")
        
        gpu = self.gpus[gpu_id]
        
        if gpu["memory_free"] < size_mb:
            raise RuntimeError(f"Not enough memory on GPU {gpu_id}. "
                             f"Requested: {size_mb}MB, Available: {gpu['memory_free']}MB")
        
        allocation_id = f"alloc_{len(self.memory_allocations)}"
        
        self.memory_allocations[allocation_id] = {
            "gpu_id": gpu_id,
            "size_mb": size_mb,
            "process_name": process_name,
            "allocated_at": datetime.now(timezone.utc)
        }
        
        # Update GPU memory usage
        gpu["memory_used"] += size_mb
        gpu["memory_free"] -= size_mb
        
        # Add process
        gpu["processes"].append({
            "name": process_name,
            "memory_mb": size_mb,
            "allocation_id": allocation_id
        })
        
        return allocation_id
    
    def free_memory(self, allocation_id: str):
        """Mock GPU memory deallocation."""
        if allocation_id not in self.memory_allocations:
            raise ValueError(f"Allocation {allocation_id} not found")
        
        allocation = self.memory_allocations[allocation_id]
        gpu_id = allocation["gpu_id"]
        size_mb = allocation["size_mb"]
        
        # Update GPU memory usage
        gpu = self.gpus[gpu_id]
        gpu["memory_used"] -= size_mb
        gpu["memory_free"] += size_mb
        
        # Remove process
        gpu["processes"] = [
            p for p in gpu["processes"] 
            if p["allocation_id"] != allocation_id
        ]
        
        # Remove allocation
        del self.memory_allocations[allocation_id]
    
    def get_memory_usage(self, gpu_id: Optional[int] = None) -> Dict:
        """Get GPU memory usage."""
        if gpu_id is not None:
            gpu = self.gpus[gpu_id]
            return {
                "gpu_id": gpu_id,
                "total_mb": gpu["memory_total"],
                "used_mb": gpu["memory_used"],
                "free_mb": gpu["memory_free"],
                "usage_percent": (gpu["memory_used"] / gpu["memory_total"]) * 100
            }
        else:
            return {
                gpu_id: self.get_memory_usage(gpu_id)
                for gpu_id in self.gpus.keys()
            }
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start GPU monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, interval: float):
        """GPU monitoring loop."""
        while self._monitoring:
            # Simulate changing GPU metrics
            for gpu in self.gpus.values():
                # Simulate utilization changes
                gpu["utilization"] = max(0, min(100, 
                    gpu["utilization"] + random.randint(-10, 10)
                ))
                
                # Simulate temperature changes
                gpu["temperature"] = max(30, min(90,
                    gpu["temperature"] + random.randint(-5, 5)
                ))
                
                # Simulate power usage changes
                gpu["power_usage"] = max(0, min(300,
                    gpu["power_usage"] + random.randint(-20, 20)
                ))
            
            time.sleep(interval)
    
    def set_gpu_availability(self, gpu_id: int, available: bool):
        """Set GPU availability for testing failures."""
        if gpu_id in self.gpus:
            self.gpus[gpu_id]["available"] = available


class MockResourceMonitor:
    """Mock system resource monitor."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alerts = []
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "gpu_memory_percent": 90
        }
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "cpu": {
                "percent": random.uniform(10, 95),
                "cores": psutil.cpu_count(),
                "frequency": random.uniform(2000, 4000)  # MHz
            },
            "memory": {
                "total_gb": 32,
                "used_gb": random.uniform(8, 28),
                "percent": random.uniform(25, 90),
                "available_gb": random.uniform(4, 24)
            },
            "disk": {
                "total_gb": 1000,
                "used_gb": random.uniform(100, 800),
                "percent": random.uniform(10, 80),
                "free_gb": random.uniform(200, 900)
            },
            "network": {
                "bytes_sent": random.randint(1000000, 10000000),
                "bytes_recv": random.randint(1000000, 10000000),
                "packets_sent": random.randint(1000, 10000),
                "packets_recv": random.randint(1000, 10000)
            },
            "processes": {
                "total": random.randint(200, 500),
                "running": random.randint(5, 50),
                "sleeping": random.randint(150, 450)
            }
        }
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
    
    def _monitoring_loop(self, interval: float):
        """Resource monitoring loop."""
        while self.monitoring:
            metrics = self.get_current_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Check thresholds and generate alerts
            self._check_thresholds(metrics)
            
            time.sleep(interval)
    
    def _check_thresholds(self, metrics: Dict):
        """Check if metrics exceed thresholds."""
        alerts = []
        
        if metrics["cpu"]["percent"] > self.thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu_high",
                "message": f"CPU usage is {metrics['cpu']['percent']:.1f}%",
                "severity": "warning",
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["memory"]["percent"] > self.thresholds["memory_percent"]:
            alerts.append({
                "type": "memory_high",
                "message": f"Memory usage is {metrics['memory']['percent']:.1f}%",
                "severity": "warning",
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["disk"]["percent"] > self.thresholds["disk_percent"]:
            alerts.append({
                "type": "disk_high",
                "message": f"Disk usage is {metrics['disk']['percent']:.1f}%",
                "severity": "critical",
                "timestamp": metrics["timestamp"]
            })
        
        self.alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics["timestamp"] > cutoff_time
        ]
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts for specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]
    
    def set_thresholds(self, thresholds: Dict):
        """Set monitoring thresholds."""
        self.thresholds.update(thresholds)


class MockConfigValidator:
    """Mock configuration validator."""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_history = []
        
    def add_validation_rule(self, rule_name: str, rule_func):
        """Add validation rule."""
        self.validation_rules[rule_name] = rule_func
    
    def validate_config(self, config: Dict, rule_names: Optional[List[str]] = None) -> Dict:
        """Validate configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now(timezone.utc)
        }
        
        rules_to_check = rule_names or list(self.validation_rules.keys())
        
        for rule_name in rules_to_check:
            if rule_name not in self.validation_rules:
                continue
            
            try:
                rule_func = self.validation_rules[rule_name]
                rule_result = rule_func(config)
                
                if not rule_result.get("valid", True):
                    validation_result["valid"] = False
                    validation_result["errors"].extend(
                        rule_result.get("errors", [])
                    )
                
                validation_result["warnings"].extend(
                    rule_result.get("warnings", [])
                )
                
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Rule '{rule_name}' failed: {str(e)}"
                )
        
        self.validation_history.append(validation_result.copy())
        return validation_result
    
    def get_validation_history(self) -> List[Dict]:
        """Get validation history."""
        return self.validation_history.copy()


class MockSecretManager:
    """Mock secret management system."""
    
    def __init__(self):
        self.secrets = {}
        self.access_log = []
        self._encryption_key = "mock_encryption_key_12345"
        
    def store_secret(self, secret_name: str, secret_value: str, metadata: Optional[Dict] = None):
        """Store a secret."""
        self.secrets[secret_name] = {
            "value": self._encrypt_value(secret_value),
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "last_accessed": None,
            "access_count": 0
        }
        
        self.access_log.append({
            "action": "store",
            "secret_name": secret_name,
            "timestamp": datetime.now(timezone.utc)
        })
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve a secret."""
        if secret_name not in self.secrets:
            return None
        
        secret = self.secrets[secret_name]
        secret["last_accessed"] = datetime.now(timezone.utc)
        secret["access_count"] += 1
        
        self.access_log.append({
            "action": "retrieve",
            "secret_name": secret_name,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return self._decrypt_value(secret["value"])
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret."""
        if secret_name not in self.secrets:
            return False
        
        del self.secrets[secret_name]
        
        self.access_log.append({
            "action": "delete",
            "secret_name": secret_name,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return True
    
    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self.secrets.keys())
    
    def get_secret_metadata(self, secret_name: str) -> Optional[Dict]:
        """Get secret metadata without retrieving value."""
        if secret_name not in self.secrets:
            return None
        
        secret = self.secrets[secret_name]
        return {
            "name": secret_name,
            "created_at": secret["created_at"],
            "last_accessed": secret["last_accessed"],
            "access_count": secret["access_count"],
            "metadata": secret["metadata"]
        }
    
    def rotate_secret(self, secret_name: str, new_value: str):
        """Rotate a secret value."""
        if secret_name not in self.secrets:
            raise ValueError(f"Secret '{secret_name}' not found")
        
        old_metadata = self.secrets[secret_name]["metadata"]
        self.store_secret(secret_name, new_value, old_metadata)
        
        self.access_log.append({
            "action": "rotate",
            "secret_name": secret_name,
            "timestamp": datetime.now(timezone.utc)
        })
    
    def _encrypt_value(self, value: str) -> str:
        """Mock encryption (not secure - for testing only)."""
        # Simple XOR encryption for testing
        encrypted = ""
        for i, char in enumerate(value):
            key_char = self._encryption_key[i % len(self._encryption_key)]
            encrypted += chr(ord(char) ^ ord(key_char))
        return encrypted
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Mock decryption."""
        # XOR decryption (same as encryption)
        return self._encrypt_value(encrypted_value)


class MockEmailService:
    """Mock email service for notifications."""
    
    def __init__(self):
        self.sent_emails = []
        self.email_templates = {}
        self.delivery_delay = 0.1  # Mock delivery delay
        self._failure_rate = 0.0
        
    def send_email(self, to_address: str, subject: str, body: str, 
                   from_address: str = "noreply@example.com",
                   html_body: Optional[str] = None,
                   attachments: Optional[List[Dict]] = None):
        """Send email."""
        
        # Simulate delivery delay
        time.sleep(self.delivery_delay)
        
        # Simulate failures
        if random.random() < self._failure_rate:
            raise Exception("Email delivery failed")
        
        email = {
            "id": f"email_{len(self.sent_emails)}",
            "to": to_address,
            "from": from_address,
            "subject": subject,
            "body": body,
            "html_body": html_body,
            "attachments": attachments or [],
            "sent_at": datetime.now(timezone.utc),
            "status": "sent"
        }
        
        self.sent_emails.append(email)
        return email["id"]
    
    def send_template_email(self, to_address: str, template_name: str, 
                           template_data: Dict):
        """Send email using template."""
        if template_name not in self.email_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.email_templates[template_name]
        
        # Simple template substitution
        subject = template["subject"].format(**template_data)
        body = template["body"].format(**template_data)
        html_body = template.get("html_body", "").format(**template_data) if template.get("html_body") else None
        
        return self.send_email(to_address, subject, body, html_body=html_body)
    
    def add_template(self, template_name: str, subject: str, body: str, html_body: Optional[str] = None):
        """Add email template."""
        self.email_templates[template_name] = {
            "subject": subject,
            "body": body,
            "html_body": html_body
        }
    
    def get_sent_emails(self, hours: int = 24) -> List[Dict]:
        """Get sent emails from last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            email for email in self.sent_emails
            if email["sent_at"] > cutoff_time
        ]
    
    def set_failure_rate(self, rate: float):
        """Set email failure rate for testing."""
        self._failure_rate = max(0.0, min(1.0, rate))


class MockSlackNotifier:
    """Mock Slack notification service."""
    
    def __init__(self):
        self.sent_messages = []
        self.channels = {}
        self.webhooks = {}
        self._failure_rate = 0.0
        
    def send_message(self, channel: str, text: str, username: str = "Bot",
                    attachments: Optional[List[Dict]] = None):
        """Send Slack message."""
        
        # Simulate failures
        if random.random() < self._failure_rate:
            raise Exception("Slack message delivery failed")
        
        message = {
            "id": f"msg_{len(self.sent_messages)}",
            "channel": channel,
            "text": text,
            "username": username,
            "attachments": attachments or [],
            "sent_at": datetime.now(timezone.utc),
            "status": "sent"
        }
        
        self.sent_messages.append(message)
        return message["id"]
    
    def send_alert(self, severity: str, title: str, message: str, 
                   channel: str = "#alerts"):
        """Send alert message."""
        color_map = {
            "info": "good",
            "warning": "warning", 
            "error": "danger",
            "critical": "danger"
        }
        
        attachment = {
            "color": color_map.get(severity, "good"),
            "title": title,
            "text": message,
            "fields": [
                {
                    "title": "Severity",
                    "value": severity.upper(),
                    "short": True
                },
                {
                    "title": "Timestamp",
                    "value": datetime.now(timezone.utc).isoformat(),
                    "short": True
                }
            ]
        }
        
        return self.send_message(
            channel=channel,
            text=f"Alert: {title}",
            attachments=[attachment]
        )
    
    def add_webhook(self, name: str, url: str):
        """Add webhook configuration."""
        self.webhooks[name] = url
    
    def get_messages(self, channel: Optional[str] = None, hours: int = 24) -> List[Dict]:
        """Get sent messages."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        messages = [
            msg for msg in self.sent_messages
            if msg["sent_at"] > cutoff_time
        ]
        
        if channel:
            messages = [msg for msg in messages if msg["channel"] == channel]
        
        return messages
    
    def set_failure_rate(self, rate: float):
        """Set failure rate for testing."""
        self._failure_rate = max(0.0, min(1.0, rate))


class MockPrometheusCollector:
    """Mock Prometheus metrics collector."""
    
    def __init__(self):
        self.metrics = {}
        self.collection_history = []
        
    def counter(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        """Create counter metric."""
        if name not in self.metrics:
            self.metrics[name] = {
                "type": "counter",
                "description": description,
                "labels": labels or [],
                "values": {},
                "total": 0
            }
        return MockPrometheusCounter(self, name)
    
    def gauge(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        """Create gauge metric."""
        if name not in self.metrics:
            self.metrics[name] = {
                "type": "gauge",
                "description": description,
                "labels": labels or [],
                "values": {},
                "current": 0
            }
        return MockPrometheusGauge(self, name)
    
    def histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        """Create histogram metric."""
        if name not in self.metrics:
            self.metrics[name] = {
                "type": "histogram",
                "description": description,
                "buckets": buckets or [0.1, 0.5, 1.0, 5.0, 10.0],
                "observations": [],
                "count": 0,
                "sum": 0
            }
        return MockPrometheusHistogram(self, name)
    
    def collect_metrics(self) -> str:
        """Collect all metrics in Prometheus format."""
        output = []
        
        for name, metric in self.metrics.items():
            output.append(f"# HELP {name} {metric['description']}")
            output.append(f"# TYPE {name} {metric['type']}")
            
            if metric["type"] == "counter":
                if metric["values"]:
                    for labels, value in metric["values"].items():
                        output.append(f"{name}{{{labels}}} {value}")
                else:
                    output.append(f"{name} {metric['total']}")
            
            elif metric["type"] == "gauge":
                if metric["values"]:
                    for labels, value in metric["values"].items():
                        output.append(f"{name}{{{labels}}} {value}")
                else:
                    output.append(f"{name} {metric['current']}")
            
            elif metric["type"] == "histogram":
                # Histogram buckets
                for bucket in metric["buckets"]:
                    count = len([obs for obs in metric["observations"] if obs <= bucket])
                    output.append(f"{name}_bucket{{le=\"{bucket}\"}} {count}")
                output.append(f"{name}_bucket{{le=\"+Inf\"}} {metric['count']}")
                output.append(f"{name}_count {metric['count']}")
                output.append(f"{name}_sum {metric['sum']}")
        
        metrics_text = "\n".join(output)
        
        # Store collection history
        self.collection_history.append({
            "timestamp": datetime.now(timezone.utc),
            "metrics_count": len(self.metrics),
            "output_size": len(metrics_text)
        })
        
        return metrics_text
    
    def get_metric_value(self, name: str, labels: Optional[str] = None):
        """Get current metric value."""
        if name not in self.metrics:
            return None
        
        metric = self.metrics[name]
        
        if labels and labels in metric.get("values", {}):
            return metric["values"][labels]
        elif metric["type"] == "counter":
            return metric["total"]
        elif metric["type"] == "gauge":
            return metric["current"]
        elif metric["type"] == "histogram":
            return {
                "count": metric["count"],
                "sum": metric["sum"],
                "buckets": metric["buckets"]
            }
        
        return None


class MockPrometheusCounter:
    """Mock Prometheus counter."""
    
    def __init__(self, collector, name):
        self.collector = collector
        self.name = name
    
    def inc(self, amount: float = 1, labels: Optional[Dict] = None):
        """Increment counter."""
        metric = self.collector.metrics[self.name]
        
        if labels:
            labels_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            if labels_str not in metric["values"]:
                metric["values"][labels_str] = 0
            metric["values"][labels_str] += amount
        else:
            metric["total"] += amount


class MockPrometheusGauge:
    """Mock Prometheus gauge."""
    
    def __init__(self, collector, name):
        self.collector = collector
        self.name = name
    
    def set(self, value: float, labels: Optional[Dict] = None):
        """Set gauge value."""
        metric = self.collector.metrics[self.name]
        
        if labels:
            labels_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            metric["values"][labels_str] = value
        else:
            metric["current"] = value
    
    def inc(self, amount: float = 1, labels: Optional[Dict] = None):
        """Increment gauge."""
        current = self.get_value(labels)
        self.set(current + amount, labels)
    
    def dec(self, amount: float = 1, labels: Optional[Dict] = None):
        """Decrement gauge."""
        current = self.get_value(labels)
        self.set(current - amount, labels)
    
    def get_value(self, labels: Optional[Dict] = None) -> float:
        """Get current gauge value."""
        metric = self.collector.metrics[self.name]
        
        if labels:
            labels_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            return metric["values"].get(labels_str, 0)
        else:
            return metric["current"]


class MockPrometheusHistogram:
    """Mock Prometheus histogram."""
    
    def __init__(self, collector, name):
        self.collector = collector
        self.name = name
    
    def observe(self, value: float):
        """Observe a value."""
        metric = self.collector.metrics[self.name]
        metric["observations"].append(value)
        metric["count"] += 1
        metric["sum"] += value


class MockGrafanaClient:
    """Mock Grafana client for dashboards."""
    
    def __init__(self):
        self.dashboards = {}
        self.datasources = {}
        self.alerts = {}
        
    def create_dashboard(self, dashboard_config: Dict) -> str:
        """Create dashboard."""
        dashboard_id = f"dashboard_{len(self.dashboards)}"
        
        self.dashboards[dashboard_id] = {
            "id": dashboard_id,
            "config": dashboard_config,
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc)
        }
        
        return dashboard_id
    
    def update_dashboard(self, dashboard_id: str, dashboard_config: Dict):
        """Update dashboard."""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        self.dashboards[dashboard_id]["config"] = dashboard_config
        self.dashboards[dashboard_id]["last_updated"] = datetime.now(timezone.utc)
    
    def delete_dashboard(self, dashboard_id: str):
        """Delete dashboard."""
        if dashboard_id in self.dashboards:
            del self.dashboards[dashboard_id]
    
    def add_datasource(self, name: str, datasource_config: Dict):
        """Add data source."""
        self.datasources[name] = {
            "name": name,
            "config": datasource_config,
            "created_at": datetime.now(timezone.utc)
        }
    
    def create_alert(self, alert_config: Dict) -> str:
        """Create alert."""
        alert_id = f"alert_{len(self.alerts)}"
        
        self.alerts[alert_id] = {
            "id": alert_id,
            "config": alert_config,
            "created_at": datetime.now(timezone.utc),
            "status": "active"
        }
        
        return alert_id
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dict]:
        """Get dashboard configuration."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict]:
        """List all dashboards."""
        return [
            {
                "id": dashboard_id,
                "title": dashboard["config"].get("title", "Untitled"),
                "created_at": dashboard["created_at"],
                "last_updated": dashboard["last_updated"]
            }
            for dashboard_id, dashboard in self.dashboards.items()
        ]