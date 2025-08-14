"""Database models for predictions and feedback storage."""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()


class PredictionRecord(Base):
    """Model for storing prediction records."""
    
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=False, index=True)  # For deduplication
    
    # Prediction results
    decision = Column(String(50))
    confidence = Column(Float)
    abstained = Column(Boolean, default=False)
    tier = Column(Integer)
    
    # Model information
    model_version = Column(String(100))
    voter_decisions = Column(JSON)  # Store all voter outputs
    
    # Performance metrics
    latency_ms = Column(Float)
    cost_cents = Column(Float)
    cached = Column(Boolean, default=False)
    
    # Metadata
    input_metadata = Column(JSON)
    slice_tags = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime)
    
    # True label (when feedback is provided)
    true_label = Column(String(50))
    feedback_received_at = Column(DateTime)


class FeedbackRecord(Base):
    """Model for storing human feedback."""
    
    __tablename__ = 'feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), nullable=False, index=True)
    
    # Feedback details
    true_label = Column(String(50), nullable=False)
    feedback_score = Column(Float)  # 0-1 quality score
    notes = Column(Text)
    
    # Correctness analysis
    was_correct = Column(Boolean)
    error_type = Column(String(50))  # 'false_positive', 'false_negative', etc.
    
    # Metadata
    reviewer_id = Column(String(100))
    review_time_seconds = Column(Float)
    confidence_in_feedback = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ModelPerformance(Base):
    """Model for tracking model performance over time."""
    
    __tablename__ = 'model_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Model identification
    model_type = Column(String(50), nullable=False)  # 'voter', 'stacker', 'ensemble'
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(100), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    f1_weighted = Column(Float)
    f1_macro = Column(Float)
    precision_macro = Column(Float)
    recall_macro = Column(Float)
    
    # Per-class metrics (JSON)
    per_class_metrics = Column(JSON)
    
    # Operational metrics
    abstention_rate = Column(Float)
    avg_latency_ms = Column(Float)
    avg_cost_cents = Column(Float)
    
    # Evaluation details
    n_samples = Column(Integer)
    evaluation_period_start = Column(DateTime)
    evaluation_period_end = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class DataDrift(Base):
    """Model for storing data drift detection results."""
    
    __tablename__ = 'data_drift'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Drift detection details
    feature_name = Column(String(100), nullable=False)
    drift_type = Column(String(50), nullable=False)  # 'covariate', 'concept', 'prediction'
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float)
    is_drift = Column(Boolean, nullable=False)
    severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Detection configuration
    detection_method = Column(String(50))
    threshold = Column(Float)
    
    # Sample information
    reference_window_size = Column(Integer)
    detection_window_size = Column(Integer)
    
    # Additional details
    details = Column(JSON)
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)


class SystemHealth(Base):
    """Model for storing system health metrics."""
    
    __tablename__ = 'system_health'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # System metrics
    active_requests = Column(Integer)
    queue_length = Column(Integer)
    cache_hit_rate = Column(Float)
    error_rate = Column(Float)
    
    # Performance metrics
    avg_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    
    # Resource usage
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    disk_usage_percent = Column(Float)
    
    # Service availability
    services_up = Column(JSON)  # Status of each service
    
    # Timestamps
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)


class ExperimentRun(Base):
    """Model for storing experiment run information."""
    
    __tablename__ = 'experiment_runs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Experiment details
    experiment_name = Column(String(200), nullable=False)
    run_name = Column(String(200), nullable=False)
    mlflow_run_id = Column(String(100), unique=True)
    
    # Configuration
    config = Column(JSON)
    hyperparameters = Column(JSON)
    
    # Results
    metrics = Column(JSON)
    model_artifacts = Column(JSON)
    
    # Status
    status = Column(String(50), default='running')  # 'running', 'completed', 'failed'
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Additional metadata
    git_commit = Column(String(40))
    environment = Column(String(50))
    notes = Column(Text)


# Database connection and session management
class DatabaseManager:
    """Manage database connections and sessions."""
    
    def __init__(self, database_url: str):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
        
    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception:
            return False


# Database operations
class DatabaseOperations:
    """Database operations for the application."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db_manager = db_manager
    
    def store_prediction(
        self,
        request_id: str,
        text: str,
        prediction_result: dict,
        metadata: dict = None
    ):
        """Store prediction result."""
        import hashlib
        
        with self.db_manager.get_session() as session:
            # Create text hash for deduplication
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            prediction = PredictionRecord(
                request_id=request_id,
                text=text,
                text_hash=text_hash,
                decision=prediction_result.get('decision'),
                confidence=prediction_result.get('confidence'),
                abstained=prediction_result.get('abstain', False),
                tier=prediction_result.get('tier'),
                model_version=prediction_result.get('model_version'),
                voter_decisions=prediction_result.get('voter_outputs'),
                latency_ms=prediction_result.get('latency_ms'),
                cost_cents=prediction_result.get('cost_cents'),
                cached=prediction_result.get('cached', False),
                input_metadata=metadata,
                processed_at=datetime.utcnow()
            )
            
            session.add(prediction)
            session.commit()
    
    def store_feedback(
        self,
        request_id: str,
        true_label: str,
        feedback_score: float = None,
        notes: str = None,
        reviewer_id: str = None
    ):
        """Store human feedback."""
        with self.db_manager.get_session() as session:
            # Get original prediction
            prediction = session.query(PredictionRecord).filter_by(
                request_id=request_id
            ).first()
            
            if prediction:
                # Update prediction with true label
                prediction.true_label = true_label
                prediction.feedback_received_at = datetime.utcnow()
                
                # Determine correctness
                was_correct = prediction.decision == true_label and not prediction.abstained
                
                # Determine error type
                error_type = None
                if not was_correct:
                    if prediction.abstained:
                        error_type = 'abstention'
                    elif prediction.decision != true_label:
                        error_type = 'misclassification'
            else:
                was_correct = None
                error_type = 'no_prediction_found'
            
            # Create feedback record
            feedback = FeedbackRecord(
                request_id=request_id,
                true_label=true_label,
                feedback_score=feedback_score,
                notes=notes,
                was_correct=was_correct,
                error_type=error_type,
                reviewer_id=reviewer_id
            )
            
            session.add(feedback)
            session.commit()
    
    def get_performance_metrics(
        self,
        time_window_hours: int = 24,
        model_name: str = None
    ) -> dict:
        """Get performance metrics for a time window."""
        from sqlalchemy import func
        
        with self.db_manager.get_session() as session:
            # Base query
            query = session.query(PredictionRecord).filter(
                PredictionRecord.created_at >= datetime.utcnow() - timedelta(hours=time_window_hours)
            )
            
            if model_name:
                query = query.filter(PredictionRecord.model_version.contains(model_name))
            
            predictions = query.all()
            
            if not predictions:
                return {}
            
            # Calculate metrics
            total = len(predictions)
            abstained = sum(1 for p in predictions if p.abstained)
            
            # Get predictions with feedback
            with_feedback = [p for p in predictions if p.true_label]
            correct = sum(1 for p in with_feedback if p.decision == p.true_label and not p.abstained)
            
            metrics = {
                'total_predictions': total,
                'abstention_rate': abstained / total,
                'accuracy': correct / len(with_feedback) if with_feedback else None,
                'avg_latency_ms': sum(p.latency_ms or 0 for p in predictions) / total,
                'avg_cost_cents': sum(p.cost_cents or 0 for p in predictions) / total,
                'cache_hit_rate': sum(1 for p in predictions if p.cached) / total
            }
            
            return metrics
    
    def get_drift_alerts(self, hours: int = 24) -> list:
        """Get recent drift alerts."""
        with self.db_manager.get_session() as session:
            drifts = session.query(DataDrift).filter(
                DataDrift.detected_at >= datetime.utcnow() - timedelta(hours=hours),
                DataDrift.is_drift == True,
                DataDrift.severity.in_(['high', 'critical'])
            ).order_by(DataDrift.detected_at.desc()).all()
            
            return [
                {
                    'feature': d.feature_name,
                    'type': d.drift_type,
                    'severity': d.severity,
                    'score': d.drift_score,
                    'detected_at': d.detected_at
                }
                for d in drifts
            ]
    
    def get_error_analysis(self, hours: int = 24) -> dict:
        """Get error analysis for recent predictions."""
        with self.db_manager.get_session() as session:
            feedback_records = session.query(FeedbackRecord).filter(
                FeedbackRecord.created_at >= datetime.utcnow() - timedelta(hours=hours)
            ).all()
            
            if not feedback_records:
                return {}
            
            total = len(feedback_records)
            errors_by_type = {}
            
            for record in feedback_records:
                if not record.was_correct:
                    error_type = record.error_type or 'unknown'
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            
            return {
                'total_feedback': total,
                'error_count': sum(errors_by_type.values()),
                'error_rate': sum(errors_by_type.values()) / total,
                'errors_by_type': errors_by_type
            }