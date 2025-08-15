"""
Mock implementations for external services.

This module provides comprehensive mocking for external APIs and services
including HuggingFace, OpenAI, databases, cloud services, and more.
"""

import json
import time
import random
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Iterator
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sqlite3
import tempfile
import shutil


class MockHuggingFaceAPI:
    """Mock HuggingFace API for models and datasets."""
    
    def __init__(self):
        self.models_cache = {}
        self.datasets_cache = {}
        self.download_history = []
        self._simulate_network_delay = True
        self._failure_rate = 0.0
        
    def from_pretrained(self, model_name: str, **kwargs):
        """Mock model loading from HuggingFace."""
        if self._simulate_network_delay:
            time.sleep(random.uniform(0.1, 0.5))
        
        if random.random() < self._failure_rate:
            raise ConnectionError(f"Failed to download model {model_name}")
        
        # Create mock model
        mock_model = MockTransformerModel(model_name)
        self.models_cache[model_name] = mock_model
        self.download_history.append({
            "type": "model",
            "name": model_name,
            "timestamp": datetime.now(timezone.utc),
            "kwargs": kwargs
        })
        
        return mock_model
    
    def load_dataset(self, dataset_name: str, **kwargs):
        """Mock dataset loading from HuggingFace."""
        if self._simulate_network_delay:
            time.sleep(random.uniform(0.1, 0.3))
        
        if random.random() < self._failure_rate:
            raise ConnectionError(f"Failed to download dataset {dataset_name}")
        
        # Create mock dataset
        mock_dataset = MockTrainingDataset(dataset_name, size=kwargs.get("size", 1000))
        self.datasets_cache[dataset_name] = mock_dataset
        self.download_history.append({
            "type": "dataset",
            "name": dataset_name,
            "timestamp": datetime.now(timezone.utc),
            "kwargs": kwargs
        })
        
        return mock_dataset
    
    def push_to_hub(self, model, repo_name: str, **kwargs):
        """Mock pushing model to HuggingFace Hub."""
        if self._simulate_network_delay:
            time.sleep(random.uniform(0.5, 2.0))
        
        if random.random() < self._failure_rate:
            raise ConnectionError(f"Failed to push to {repo_name}")
        
        return {
            "repo_name": repo_name,
            "url": f"https://huggingface.co/{repo_name}",
            "commit_sha": f"mock_sha_{random.randint(1000000, 9999999)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def list_models(self, filter_name: str = None, **kwargs):
        """Mock listing models from HuggingFace."""
        mock_models = [
            {"modelId": "bert-base-uncased", "downloads": 50000},
            {"modelId": "gpt2", "downloads": 75000},
            {"modelId": "distilbert-base-uncased", "downloads": 30000},
            {"modelId": "roberta-base", "downloads": 40000}
        ]
        
        if filter_name:
            mock_models = [m for m in mock_models if filter_name in m["modelId"]]
        
        return mock_models
    
    def set_failure_rate(self, rate: float):
        """Set the failure rate for testing error conditions."""
        self._failure_rate = max(0.0, min(1.0, rate))
    
    def disable_network_delay(self):
        """Disable network delay simulation for faster tests."""
        self._simulate_network_delay = False


class MockOpenAIAPI:
    """Mock OpenAI API for completions and embeddings."""
    
    def __init__(self):
        self.api_calls = []
        self.usage_stats = {"tokens": 0, "requests": 0, "cost": 0.0}
        self._failure_rate = 0.0
        self._rate_limit_enabled = False
        self._rate_limit_calls = 0
        self._max_calls_per_minute = 60
        
    def chat_completions_create(self, model: str, messages: List[Dict], **kwargs):
        """Mock OpenAI chat completions."""
        self._check_rate_limit()
        
        if random.random() < self._failure_rate:
            raise Exception("OpenAI API error")
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate mock response
        response_text = self._generate_mock_response(messages[-1]["content"])
        tokens_used = len(response_text.split()) + sum(len(m["content"].split()) for m in messages)
        
        response = {
            "id": f"chatcmpl-{random.randint(1000000, 9999999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m["content"].split()) for m in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": tokens_used
            }
        }
        
        # Track usage
        self.api_calls.append({
            "model": model,
            "messages": messages,
            "response": response,
            "timestamp": datetime.now(timezone.utc),
            "kwargs": kwargs
        })
        
        self.usage_stats["tokens"] += tokens_used
        self.usage_stats["requests"] += 1
        self.usage_stats["cost"] += tokens_used * 0.0001  # Mock cost calculation
        
        return response
    
    def embeddings_create(self, model: str, input_texts: Union[str, List[str]], **kwargs):
        """Mock OpenAI embeddings creation."""
        self._check_rate_limit()
        
        if random.random() < self._failure_rate:
            raise Exception("OpenAI API error")
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        # Generate mock embeddings
        embeddings = []
        for text in input_texts:
            # Generate deterministic embeddings based on text hash
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(1536)]  # OpenAI embedding size
            embeddings.append(embedding)
        
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": i
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in input_texts),
                "total_tokens": sum(len(text.split()) for text in input_texts)
            }
        }
        
        # Track usage
        self.api_calls.append({
            "type": "embeddings",
            "model": model,
            "input_texts": input_texts,
            "response": response,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return response
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        responses = [
            "This is a mock response to your query about: " + prompt[:50] + "...",
            "Based on the input provided, here's a detailed analysis...",
            "Thank you for your question. The answer depends on several factors...",
            "I understand you're asking about this topic. Let me provide some insights...",
            "This is an interesting question that requires careful consideration..."
        ]
        
        base_response = random.choice(responses)
        
        # Add some context-specific responses
        if "code" in prompt.lower():
            base_response += "\n\nHere's a code example:\n```python\ndef example_function():\n    return 'mock code'\n```"
        elif "explain" in prompt.lower():
            base_response += "\n\nTo explain this concept:\n1. First point\n2. Second point\n3. Third point"
        
        return base_response
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        if not self._rate_limit_enabled:
            return
        
        current_time = time.time()
        
        # Reset counter every minute
        if not hasattr(self, '_last_reset') or current_time - self._last_reset > 60:
            self._rate_limit_calls = 0
            self._last_reset = current_time
        
        if self._rate_limit_calls >= self._max_calls_per_minute:
            raise Exception("Rate limit exceeded")
        
        self._rate_limit_calls += 1
    
    def enable_rate_limiting(self, max_calls_per_minute: int = 60):
        """Enable rate limiting for testing."""
        self._rate_limit_enabled = True
        self._max_calls_per_minute = max_calls_per_minute
    
    def set_failure_rate(self, rate: float):
        """Set the failure rate for testing error conditions."""
        self._failure_rate = max(0.0, min(1.0, rate))


class MockDatabaseConnection:
    """Mock database connection for testing."""
    
    def __init__(self, db_type: str = "sqlite"):
        self.db_type = db_type
        self.is_connected = False
        self.transactions = []
        self.tables = {}
        self._temp_db = None
        self._connection = None
        
    def connect(self):
        """Mock database connection."""
        if self.db_type == "sqlite":
            self._temp_db = tempfile.mktemp(suffix=".db")
            self._connection = sqlite3.connect(self._temp_db)
            self._connection.row_factory = sqlite3.Row
        
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Mock database disconnection."""
        if self._connection:
            self._connection.close()
        if self._temp_db and Path(self._temp_db).exists():
            Path(self._temp_db).unlink()
        
        self.is_connected = False
        return True
    
    def execute(self, query: str, params: tuple = None):
        """Mock query execution."""
        if not self.is_connected:
            raise Exception("Database not connected")
        
        transaction = {
            "query": query,
            "params": params,
            "timestamp": datetime.now(timezone.utc),
            "type": self._get_query_type(query)
        }
        
        self.transactions.append(transaction)
        
        if self._connection:
            cursor = self._connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if transaction["type"] in ["SELECT"]:
                return cursor.fetchall()
            else:
                self._connection.commit()
                return cursor.rowcount
        
        # Mock response for non-SQLite databases
        if transaction["type"] == "SELECT":
            return self._generate_mock_select_result(query)
        else:
            return 1  # Mock affected rows
    
    def execute_many(self, query: str, params_list: List[tuple]):
        """Mock batch query execution."""
        if not self.is_connected:
            raise Exception("Database not connected")
        
        results = []
        for params in params_list:
            result = self.execute(query, params)
            results.append(result)
        
        return results
    
    def begin_transaction(self):
        """Mock transaction begin."""
        self.transactions.append({
            "type": "BEGIN_TRANSACTION",
            "timestamp": datetime.now(timezone.utc)
        })
    
    def commit_transaction(self):
        """Mock transaction commit."""
        self.transactions.append({
            "type": "COMMIT_TRANSACTION",
            "timestamp": datetime.now(timezone.utc)
        })
        
        if self._connection:
            self._connection.commit()
    
    def rollback_transaction(self):
        """Mock transaction rollback."""
        self.transactions.append({
            "type": "ROLLBACK_TRANSACTION",
            "timestamp": datetime.now(timezone.utc)
        })
        
        if self._connection:
            self._connection.rollback()
    
    def _get_query_type(self, query: str) -> str:
        """Determine query type from SQL."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        elif query_upper.startswith("CREATE"):
            return "CREATE"
        elif query_upper.startswith("DROP"):
            return "DROP"
        else:
            return "OTHER"
    
    def _generate_mock_select_result(self, query: str) -> List[Dict]:
        """Generate mock SELECT result."""
        # Simple mock data generation
        if "users" in query.lower():
            return [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ]
        elif "training" in query.lower():
            return [
                {"id": 1, "model_name": "test-model", "accuracy": 0.95, "loss": 0.05},
                {"id": 2, "model_name": "test-model-2", "accuracy": 0.92, "loss": 0.08}
            ]
        else:
            return [{"count": 42}]


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files = {}
        self.directories = set(["/"])
        self.operations = []
        self._temp_dir = None
    
    def create_temp_workspace(self) -> Path:
        """Create temporary workspace for testing."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="mock_fs_"))
        return self._temp_dir
    
    def cleanup_temp_workspace(self):
        """Clean up temporary workspace."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    def read_file(self, file_path: str) -> str:
        """Mock file reading."""
        self.operations.append({
            "operation": "read",
            "path": file_path,
            "timestamp": datetime.now(timezone.utc)
        })
        
        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return self.files[file_path]
    
    def write_file(self, file_path: str, content: str):
        """Mock file writing."""
        self.operations.append({
            "operation": "write",
            "path": file_path,
            "content_length": len(content),
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Ensure directory exists
        dir_path = str(Path(file_path).parent)
        self.directories.add(dir_path)
        
        self.files[file_path] = content
    
    def delete_file(self, file_path: str):
        """Mock file deletion."""
        self.operations.append({
            "operation": "delete",
            "path": file_path,
            "timestamp": datetime.now(timezone.utc)
        })
        
        if file_path in self.files:
            del self.files[file_path]
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def list_files(self, directory: str) -> List[str]:
        """Mock directory listing."""
        self.operations.append({
            "operation": "list",
            "path": directory,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return [
            file_path for file_path in self.files.keys()
            if file_path.startswith(directory)
        ]
    
    def file_exists(self, file_path: str) -> bool:
        """Mock file existence check."""
        return file_path in self.files
    
    def get_file_size(self, file_path: str) -> int:
        """Mock file size retrieval."""
        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return len(self.files[file_path])


class MockNetworkService:
    """Mock network service for HTTP requests."""
    
    def __init__(self):
        self.requests = []
        self.responses = {}
        self._failure_rate = 0.0
        self._latency_range = (0.1, 0.5)
    
    def get(self, url: str, **kwargs):
        """Mock HTTP GET request."""
        return self._make_request("GET", url, **kwargs)
    
    def post(self, url: str, data=None, json=None, **kwargs):
        """Mock HTTP POST request."""
        return self._make_request("POST", url, data=data, json=json, **kwargs)
    
    def put(self, url: str, data=None, json=None, **kwargs):
        """Mock HTTP PUT request."""
        return self._make_request("PUT", url, data=data, json=json, **kwargs)
    
    def delete(self, url: str, **kwargs):
        """Mock HTTP DELETE request."""
        return self._make_request("DELETE", url, **kwargs)
    
    def _make_request(self, method: str, url: str, **kwargs):
        """Internal request making with mocking."""
        # Simulate network latency
        time.sleep(random.uniform(*self._latency_range))
        
        # Simulate failures
        if random.random() < self._failure_rate:
            raise ConnectionError(f"Failed to connect to {url}")
        
        request_data = {
            "method": method,
            "url": url,
            "timestamp": datetime.now(timezone.utc),
            "kwargs": kwargs
        }
        
        self.requests.append(request_data)
        
        # Generate mock response
        if url in self.responses:
            response_data = self.responses[url]
        else:
            response_data = self._generate_default_response(method, url)
        
        # Create mock response object
        mock_response = Mock()
        mock_response.status_code = response_data.get("status_code", 200)
        mock_response.json.return_value = response_data.get("json", {})
        mock_response.text = response_data.get("text", "")
        mock_response.headers = response_data.get("headers", {})
        
        return mock_response
    
    def add_response(self, url: str, response_data: Dict):
        """Add predefined response for a URL."""
        self.responses[url] = response_data
    
    def set_failure_rate(self, rate: float):
        """Set failure rate for network requests."""
        self._failure_rate = max(0.0, min(1.0, rate))
    
    def set_latency_range(self, min_latency: float, max_latency: float):
        """Set latency range for network requests."""
        self._latency_range = (min_latency, max_latency)
    
    def _generate_default_response(self, method: str, url: str) -> Dict:
        """Generate default response based on URL patterns."""
        if "api" in url:
            return {
                "status_code": 200,
                "json": {"status": "success", "data": {"mock": "response"}},
                "headers": {"Content-Type": "application/json"}
            }
        else:
            return {
                "status_code": 200,
                "text": f"Mock response for {method} {url}",
                "headers": {"Content-Type": "text/plain"}
            }


class MockCacheService:
    """Mock cache service (Redis-like)."""
    
    def __init__(self):
        self.data = {}
        self.expiry = {}
        self.operations = []
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: str):
        """Mock cache get operation."""
        self.operations.append({
            "operation": "get",
            "key": key,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Check expiry
        if key in self.expiry:
            if datetime.now(timezone.utc) > self.expiry[key]:
                self.delete(key)
                self._miss_count += 1
                return None
        
        if key in self.data:
            self._hit_count += 1
            return self.data[key]
        else:
            self._miss_count += 1
            return None
    
    def set(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """Mock cache set operation."""
        self.operations.append({
            "operation": "set",
            "key": key,
            "value_type": type(value).__name__,
            "expire_seconds": expire_seconds,
            "timestamp": datetime.now(timezone.utc)
        })
        
        self.data[key] = value
        
        if expire_seconds:
            from datetime import timedelta
            self.expiry[key] = datetime.now(timezone.utc) + timedelta(seconds=expire_seconds)
    
    def delete(self, key: str):
        """Mock cache delete operation."""
        self.operations.append({
            "operation": "delete",
            "key": key,
            "timestamp": datetime.now(timezone.utc)
        })
        
        self.data.pop(key, None)
        self.expiry.pop(key, None)
    
    def exists(self, key: str) -> bool:
        """Mock cache key existence check."""
        return self.get(key) is not None
    
    def clear(self):
        """Mock cache clear operation."""
        self.operations.append({
            "operation": "clear",
            "timestamp": datetime.now(timezone.utc)
        })
        
        self.data.clear()
        self.expiry.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "total_keys": len(self.data),
            "operations": len(self.operations)
        }


# Import mock model classes (to be defined in model_mocks.py)
class MockTransformerModel:
    """Mock transformer model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = {"model_type": "transformer", "vocab_size": 50000}
        self._parameters = {"total": 110000000}  # Mock 110M parameters
    
    def __call__(self, input_ids, **kwargs):
        """Mock forward pass."""
        batch_size, seq_len = input_ids.shape if hasattr(input_ids, 'shape') else (1, len(input_ids))
        vocab_size = self.config["vocab_size"]
        
        # Return mock logits
        import torch
        return Mock(logits=torch.randn(batch_size, seq_len, vocab_size))
    
    def save_pretrained(self, save_directory: str):
        """Mock model saving."""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        with open(Path(save_directory) / "config.json", "w") as f:
            json.dump(self.config, f)


class MockTrainingDataset:
    """Mock training dataset."""
    
    def __init__(self, name: str, size: int = 1000):
        self.name = name
        self.size = size
        self._data = None
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [2, 3, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1, 1]
        }
    
    def map(self, function, **kwargs):
        """Mock dataset mapping."""
        return self
    
    def filter(self, function, **kwargs):
        """Mock dataset filtering."""
        return self


# Additional mock services for cloud providers, experiment tracking, etc.

class MockMLFlowTracker:
    """Mock MLFlow experiment tracker."""
    
    def __init__(self):
        self.experiments = {}
        self.runs = {}
        self.artifacts = {}
    
    def create_experiment(self, name: str):
        """Mock experiment creation."""
        exp_id = f"exp_{len(self.experiments)}"
        self.experiments[exp_id] = {"name": name, "created": datetime.now(timezone.utc)}
        return exp_id
    
    def start_run(self, experiment_id: str = None):
        """Mock run start."""
        run_id = f"run_{len(self.runs)}"
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "status": "RUNNING",
            "start_time": datetime.now(timezone.utc),
            "metrics": {},
            "params": {}
        }
        return run_id
    
    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        """Mock metric logging."""
        if run_id in self.runs:
            if key not in self.runs[run_id]["metrics"]:
                self.runs[run_id]["metrics"][key] = []
            self.runs[run_id]["metrics"][key].append({
                "value": value,
                "step": step,
                "timestamp": datetime.now(timezone.utc)
            })
    
    def log_param(self, run_id: str, key: str, value: str):
        """Mock parameter logging."""
        if run_id in self.runs:
            self.runs[run_id]["params"][key] = value


class MockWandBTracker:
    """Mock Weights & Biases experiment tracker."""
    
    def __init__(self):
        self.projects = {}
        self.runs = {}
    
    def init(self, project: str, name: str = None, config: Dict = None):
        """Mock wandb init."""
        if project not in self.projects:
            self.projects[project] = {"created": datetime.now(timezone.utc), "runs": []}
        
        run_id = f"wandb_run_{len(self.runs)}"
        self.runs[run_id] = {
            "project": project,
            "name": name or f"run_{len(self.runs)}",
            "config": config or {},
            "status": "running",
            "logs": []
        }
        
        self.projects[project]["runs"].append(run_id)
        return run_id
    
    def log(self, run_id: str, data: Dict):
        """Mock wandb logging."""
        if run_id in self.runs:
            self.runs[run_id]["logs"].append({
                "data": data,
                "timestamp": datetime.now(timezone.utc)
            })


class MockS3Service:
    """Mock AWS S3 service."""
    
    def __init__(self):
        self.buckets = {}
        self.objects = {}
    
    def create_bucket(self, bucket_name: str):
        """Mock bucket creation."""
        self.buckets[bucket_name] = {
            "created": datetime.now(timezone.utc),
            "objects": []
        }
    
    def upload_file(self, file_path: str, bucket: str, key: str):
        """Mock file upload."""
        if bucket not in self.buckets:
            self.create_bucket(bucket)
        
        # Simulate file upload
        file_size = len(str(file_path)) * 100  # Mock file size
        self.objects[f"{bucket}/{key}"] = {
            "bucket": bucket,
            "key": key,
            "size": file_size,
            "uploaded": datetime.now(timezone.utc)
        }
        
        self.buckets[bucket]["objects"].append(key)
    
    def download_file(self, bucket: str, key: str, local_path: str):
        """Mock file download."""
        if f"{bucket}/{key}" not in self.objects:
            raise Exception(f"Object {key} not found in bucket {bucket}")
        
        # Simulate file download
        return {"status": "success", "local_path": local_path}


class MockGCSService:
    """Mock Google Cloud Storage service."""
    
    def __init__(self):
        self.buckets = {}
        self.blobs = {}
    
    def create_bucket(self, bucket_name: str):
        """Mock GCS bucket creation."""
        self.buckets[bucket_name] = {
            "created": datetime.now(timezone.utc),
            "blobs": []
        }
    
    def upload_blob(self, bucket_name: str, source_file: str, destination_blob: str):
        """Mock blob upload."""
        if bucket_name not in self.buckets:
            self.create_bucket(bucket_name)
        
        blob_key = f"{bucket_name}/{destination_blob}"
        self.blobs[blob_key] = {
            "bucket": bucket_name,
            "name": destination_blob,
            "size": len(str(source_file)) * 100,
            "uploaded": datetime.now(timezone.utc)
        }
        
        self.buckets[bucket_name]["blobs"].append(destination_blob)


class MockDockerService:
    """Mock Docker service for containerization."""
    
    def __init__(self):
        self.images = {}
        self.containers = {}
    
    def build_image(self, tag: str, dockerfile_path: str):
        """Mock Docker image build."""
        self.images[tag] = {
            "tag": tag,
            "dockerfile_path": dockerfile_path,
            "built": datetime.now(timezone.utc),
            "size": random.randint(100, 1000)  # Mock size in MB
        }
    
    def run_container(self, image: str, name: str = None, **kwargs):
        """Mock Docker container run."""
        container_id = f"container_{len(self.containers)}"
        self.containers[container_id] = {
            "image": image,
            "name": name or container_id,
            "status": "running",
            "started": datetime.now(timezone.utc),
            "kwargs": kwargs
        }
        return container_id
    
    def stop_container(self, container_id: str):
        """Mock container stop."""
        if container_id in self.containers:
            self.containers[container_id]["status"] = "stopped"
            self.containers[container_id]["stopped"] = datetime.now(timezone.utc)


class MockKubernetesService:
    """Mock Kubernetes service for orchestration."""
    
    def __init__(self):
        self.namespaces = {"default": {"created": datetime.now(timezone.utc)}}
        self.pods = {}
        self.services = {}
        self.deployments = {}
    
    def create_pod(self, name: str, image: str, namespace: str = "default"):
        """Mock pod creation."""
        pod_id = f"pod_{len(self.pods)}"
        self.pods[pod_id] = {
            "name": name,
            "image": image,
            "namespace": namespace,
            "status": "running",
            "created": datetime.now(timezone.utc)
        }
        return pod_id
    
    def create_deployment(self, name: str, image: str, replicas: int = 1):
        """Mock deployment creation."""
        deployment_id = f"deploy_{len(self.deployments)}"
        self.deployments[deployment_id] = {
            "name": name,
            "image": image,
            "replicas": replicas,
            "status": "running",
            "created": datetime.now(timezone.utc)
        }
        return deployment_id
    
    def scale_deployment(self, deployment_id: str, replicas: int):
        """Mock deployment scaling."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["replicas"] = replicas
            self.deployments[deployment_id]["last_scaled"] = datetime.now(timezone.utc)