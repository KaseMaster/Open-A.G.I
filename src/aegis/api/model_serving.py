"""
AEGIS Model Serving API
REST API endpoints for model inference and federated learning
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from datetime import datetime
import logging
import hashlib
import json
from pathlib import Path

from ..ml.federated_learning import (
    FederatedServer,
    FederatedClient,
    FederatedConfig,
    AggregationStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AEGIS Model Serving API",
    description="REST API for model inference and federated learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelMetadata(BaseModel):
    """Metadata for trained models"""
    model_id: str
    model_name: str
    version: str
    created_at: datetime
    accuracy: Optional[float] = None
    num_parameters: int
    framework: str = "pytorch"
    tags: Dict[str, str] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    """Request for model inference"""
    model_id: str
    input_data: List[List[float]]
    batch_size: Optional[int] = 32


class InferenceResponse(BaseModel):
    """Response from model inference"""
    model_id: str
    predictions: List[Any]
    confidence_scores: Optional[List[float]] = None
    inference_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class TrainingRequest(BaseModel):
    """Request to start federated training"""
    model_name: str
    aggregation_strategy: str = "fedavg"
    num_rounds: int = 10
    clients_per_round: int = 5
    local_epochs: int = 3
    learning_rate: float = 0.01
    config_override: Optional[Dict[str, Any]] = None


class TrainingStatus(BaseModel):
    """Status of federated training job"""
    job_id: str
    status: str
    current_round: int
    total_rounds: int
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    started_at: datetime
    updated_at: datetime


class ModelRegistry:
    """Registry for managing trained models"""
    
    def __init__(self, storage_dir: str = "models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, nn.Module] = {}
    
    def register_model(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        accuracy: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a new model"""
        model_id = hashlib.sha256(
            f"{model_name}_{version}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        num_params = sum(p.numel() for p in model.parameters())
        
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            created_at=datetime.now(),
            accuracy=accuracy,
            num_parameters=num_params,
            tags=tags or {}
        )
        
        model_path = self.storage_dir / f"{model_id}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": metadata.dict()
        }, model_path)
        
        self.models[model_id] = metadata
        self.loaded_models[model_id] = model
        
        logger.info(f"Registered model: {model_id} ({model_name} v{version})")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get a loaded model by ID"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        model_path = self.storage_dir / f"{model_id}.pt"
        if not model_path.exists():
            return None
        
        checkpoint = torch.load(model_path, map_location="cpu")
        return checkpoint
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models"""
        return list(self.models.values())
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry"""
        if model_id not in self.models:
            return False
        
        model_path = self.storage_dir / f"{model_id}.pt"
        if model_path.exists():
            model_path.unlink()
        
        del self.models[model_id]
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        
        logger.info(f"Deleted model: {model_id}")
        return True


class FederatedTrainingManager:
    """Manager for federated learning training jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.servers: Dict[str, FederatedServer] = {}
    
    def start_training(
        self,
        model: nn.Module,
        config: FederatedConfig,
        model_name: str
    ) -> str:
        """Start a new federated training job"""
        job_id = hashlib.sha256(
            f"{model_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        server = FederatedServer(
            model=model,
            config=config,
            device="cpu"
        )
        
        self.servers[job_id] = server
        self.jobs[job_id] = {
            "job_id": job_id,
            "model_name": model_name,
            "status": "running",
            "config": config,
            "started_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        logger.info(f"Started federated training job: {job_id}")
        return job_id
    
    def get_status(self, job_id: str) -> Optional[TrainingStatus]:
        """Get status of a training job"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        server = self.servers.get(job_id)
        
        current_loss = None
        current_accuracy = None
        current_round = 0
        
        if server:
            current_round = server.state.round_number
            current_loss = server.state.aggregated_loss
            current_accuracy = server.state.aggregated_accuracy
        
        return TrainingStatus(
            job_id=job_id,
            status=job["status"],
            current_round=current_round,
            total_rounds=job["config"].num_rounds,
            current_loss=current_loss,
            current_accuracy=current_accuracy,
            started_at=job["started_at"],
            updated_at=job["updated_at"]
        )
    
    def get_server(self, job_id: str) -> Optional[FederatedServer]:
        """Get federated server for a job"""
        return self.servers.get(job_id)
    
    def stop_training(self, job_id: str) -> bool:
        """Stop a training job"""
        if job_id not in self.jobs:
            return False
        
        self.jobs[job_id]["status"] = "stopped"
        self.jobs[job_id]["updated_at"] = datetime.now()
        
        logger.info(f"Stopped federated training job: {job_id}")
        return True


model_registry = ModelRegistry()
training_manager = FederatedTrainingManager()


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "AEGIS Model Serving API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "models": "/api/v1/models",
            "inference": "/api/v1/inference",
            "training": "/api/v1/training",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_registered": len(model_registry.models),
        "active_training_jobs": len([j for j in training_manager.jobs.values() if j["status"] == "running"])
    }


@app.get("/api/v1/models", response_model=List[ModelMetadata])
async def list_models():
    """List all registered models"""
    return model_registry.list_models()


@app.get("/api/v1/models/{model_id}", response_model=ModelMetadata)
async def get_model_metadata(model_id: str):
    """Get metadata for a specific model"""
    if model_id not in model_registry.models:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_registry.models[model_id]


@app.delete("/api/v1/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry"""
    if not model_registry.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": f"Model {model_id} deleted successfully"}


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Perform inference with a registered model"""
    import time
    
    model = model_registry.get_model(request.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if isinstance(model, dict):
        raise HTTPException(status_code=500, detail="Model architecture not available")
    
    start_time = time.time()
    
    try:
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.argmax(dim=1).tolist()
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                confidence_scores = torch.softmax(outputs, dim=1).max(dim=1)[0].tolist()
            else:
                confidence_scores = None
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            model_id=request.model_id,
            predictions=predictions,
            confidence_scores=confidence_scores,
            inference_time_ms=inference_time_ms
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/v1/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new federated training job"""
    try:
        strategy = AggregationStrategy(request.aggregation_strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aggregation strategy. Valid options: {[s.value for s in AggregationStrategy]}"
        )
    
    config = FederatedConfig(
        aggregation_strategy=strategy,
        num_rounds=request.num_rounds,
        clients_per_round=request.clients_per_round,
        local_epochs=request.local_epochs,
        learning_rate=request.learning_rate
    )
    
    if request.config_override:
        for key, value in request.config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    from torch import nn
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    
    job_id = training_manager.start_training(
        model=model,
        config=config,
        model_name=request.model_name
    )
    
    return {
        "job_id": job_id,
        "message": "Training job started successfully",
        "status_endpoint": f"/api/v1/training/status/{job_id}"
    }


@app.get("/api/v1/training/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get status of a training job"""
    status = training_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return status


@app.post("/api/v1/training/stop/{job_id}")
async def stop_training(job_id: str):
    """Stop a training job"""
    if not training_manager.stop_training(job_id):
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"message": f"Training job {job_id} stopped successfully"}


@app.get("/api/v1/training/metrics/{job_id}")
async def get_training_metrics(job_id: str):
    """Get training metrics for a job"""
    server = training_manager.get_server(job_id)
    if server is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return {
        "job_id": job_id,
        "current_round": server.state.round_number,
        "metrics": {
            "loss": server.state.performance_metrics.get("loss", []),
            "accuracy": server.state.performance_metrics.get("accuracy", []),
            "training_time": server.state.performance_metrics.get("training_time", [])
        },
        "client_history": server.state.client_history
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
