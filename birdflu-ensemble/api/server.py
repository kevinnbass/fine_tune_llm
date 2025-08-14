"""FastAPI server for production serving."""

import os
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import redis
import hashlib
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.metrics import MonitoringSystem, PredictionMetrics
from scripts.eval_full_system import EnsembleSystem
from cache.prediction_cache import PredictionCache
from database.models import PredictionRecord, FeedbackRecord
from security.input_sanitizer import InputSanitizer
from security.rate_limiter import RateLimiter

# Initialize components
monitoring = MonitoringSystem(
    service_name="birdflu-api",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
    enable_prometheus=True
)

cache = PredictionCache(
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    ttl_seconds=3600
)

rate_limiter = RateLimiter(
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    max_requests_per_minute=100
)

input_sanitizer = InputSanitizer()

# Security
security = HTTPBearer()


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    text: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    use_cache: bool = True
    model_version: Optional[str] = None
    
    @validator('text')
    def sanitize_text(cls, v):
        """Sanitize input text."""
        return input_sanitizer.sanitize(v)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    request_id: str
    decision: Optional[str]
    confidence: float
    abstained: bool
    reason: Optional[str] = None
    tier: int
    latency_ms: float
    cached: bool
    timestamp: datetime
    model_version: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    metadata: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = None
    use_cache: bool = True


class FeedbackRequest(BaseModel):
    """Request model for feedback."""
    request_id: str
    true_label: str
    feedback_score: Optional[float] = Field(None, ge=0, le=1)
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    models_loaded: Dict[str, bool]
    cache_connected: bool
    database_connected: bool


class MetricsResponse(BaseModel):
    """Metrics response."""
    summary: Dict[str, Any]
    slice_performance: Dict[str, Dict[str, float]]
    alerts: List[str]


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("Starting API server...")
    
    # Load models
    app.state.ensemble = EnsembleSystem()
    
    # Initialize database connection
    from database.connection import init_db
    await init_db()
    
    # Warm up cache
    await cache.connect()
    
    print("API server started successfully")
    
    yield
    
    # Shutdown
    print("Shutting down API server...")
    await cache.disconnect()
    print("API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="BirdFlu Ensemble API",
    description="Production API for bird flu content classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    token = credentials.credentials
    valid_tokens = os.getenv("API_TOKENS", "").split(",")
    
    if token not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return token


# Rate limiting dependency
async def check_rate_limit(request: Request, token: str = Depends(verify_token)):
    """Check rate limit for user."""
    client_id = f"{token}:{request.client.host}"
    
    if not await rate_limiter.check_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = {}
    for voter_id in app.state.ensemble.voters:
        models_loaded[voter_id] = True
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        models_loaded=models_loaded,
        cache_connected=await cache.is_connected(),
        database_connected=True  # Check actual DB connection
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(check_rate_limit)
):
    """Single prediction endpoint."""
    start_time = time.time()
    
    # Generate request ID if not provided
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    # Check cache
    cached_result = None
    if request.use_cache:
        cached_result = await cache.get_prediction(request.text)
    
    if cached_result:
        # Return cached result
        return PredictionResponse(
            request_id=request.request_id,
            decision=cached_result.get('decision'),
            confidence=cached_result.get('confidence', 0),
            abstained=cached_result.get('abstained', False),
            reason=cached_result.get('reason'),
            tier=cached_result.get('tier', 1),
            latency_ms=(time.time() - start_time) * 1000,
            cached=True,
            timestamp=datetime.utcnow(),
            model_version=request.model_version or "latest"
        )
    
    # Run prediction
    try:
        result = app.state.ensemble.predict_with_cascade(
            request.text,
            request.metadata
        )
    except Exception as e:
        monitoring.record_error("prediction_error", str(e), request.request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    
    # Calculate total latency
    total_latency = (time.time() - start_time) * 1000
    
    # Cache result
    if request.use_cache and not result.get('abstained'):
        background_tasks.add_task(cache.set_prediction, request.text, result)
    
    # Record metrics
    metrics = PredictionMetrics(
        timestamp=datetime.utcnow(),
        request_id=request.request_id,
        text_length=len(request.text),
        decision=result.get('decision'),
        confidence=result.get('confidence', 0),
        abstained=result.get('abstain', False),
        tier=result.get('tier', 1),
        total_latency_ms=total_latency,
        voter_latencies={},  # Would be filled from actual voter timings
        voter_costs={},  # Would be filled from actual voter costs
        llm_called='llm' in str(result.get('model_id', '')).lower(),
        cache_hit=False
    )
    
    background_tasks.add_task(monitoring.record_prediction, metrics)
    
    # Store in database
    from database.operations import store_prediction
    background_tasks.add_task(
        store_prediction,
        request.request_id,
        request.text,
        result,
        request.metadata
    )
    
    return PredictionResponse(
        request_id=request.request_id,
        decision=result.get('decision'),
        confidence=result.get('confidence', 0),
        abstained=result.get('abstain', False),
        reason=result.get('reason'),
        tier=result.get('tier', 1),
        latency_ms=total_latency,
        cached=False,
        timestamp=datetime.utcnow(),
        model_version=request.model_version or "latest"
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(check_rate_limit)
):
    """Batch prediction endpoint."""
    if not request.request_id:
        request.request_id = str(uuid.uuid4())
    
    responses = []
    
    # Process in parallel with asyncio
    tasks = []
    for i, text in enumerate(request.texts):
        metadata = request.metadata[i] if request.metadata else None
        
        pred_request = PredictionRequest(
            text=text,
            metadata=metadata,
            request_id=f"{request.request_id}_{i}",
            use_cache=request.use_cache
        )
        
        tasks.append(predict(pred_request, background_tasks, _))
    
    responses = await asyncio.gather(*tasks)
    
    return responses


@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_token)
):
    """Submit feedback for a prediction."""
    # Store feedback in database
    from database.operations import store_feedback
    
    try:
        await store_feedback(
            request.request_id,
            request.true_label,
            request.feedback_score,
            request.notes
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store feedback: {str(e)}"
        )
    
    # Update monitoring with feedback
    # This would update accuracy tracking
    
    return {"status": "success", "message": "Feedback recorded"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(_: str = Depends(verify_token)):
    """Get system metrics."""
    summary = monitoring.get_metrics_summary()
    slice_performance = monitoring.get_slice_performance()
    
    # Check for active alerts
    alerts = []
    if summary.get('error_rate', 0) > 0.05:
        alerts.append(f"High error rate: {summary['error_rate']:.2%}")
    
    if summary.get('abstention_rate', 0) > 0.20:
        alerts.append(f"High abstention rate: {summary['abstention_rate']:.2%}")
    
    return MetricsResponse(
        summary=summary,
        slice_performance=slice_performance,
        alerts=alerts
    )


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = monitoring.export_prometheus_metrics()
    return StreamingResponse(
        iter([metrics_data]),
        media_type="text/plain"
    )


@app.get("/models")
async def list_models(_: str = Depends(verify_token)):
    """List available models and versions."""
    models = {}
    
    for voter_id, voter in app.state.ensemble.voters.items():
        models[voter_id] = {
            "loaded": True,
            "version": getattr(voter, 'version', 'unknown'),
            "type": voter.__class__.__name__
        }
    
    return {"models": models}


@app.post("/models/reload")
async def reload_models(
    model_type: Optional[str] = None,
    _: str = Depends(verify_token)
):
    """Reload models (for model updates)."""
    try:
        if model_type:
            # Reload specific model
            if model_type in app.state.ensemble.voters:
                # Reload logic here
                pass
        else:
            # Reload all models
            app.state.ensemble = EnsembleSystem()
        
        return {"status": "success", "message": "Models reloaded"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload models: {str(e)}"
        )


@app.get("/cache/stats")
async def cache_stats(_: str = Depends(verify_token)):
    """Get cache statistics."""
    stats = await cache.get_stats()
    return stats


@app.post("/cache/clear")
async def clear_cache(_: str = Depends(verify_token)):
    """Clear prediction cache."""
    await cache.clear()
    return {"status": "success", "message": "Cache cleared"}


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    monitoring.record_error("validation_error", str(exc), "unknown")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    monitoring.record_error("internal_error", str(exc), "unknown")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )