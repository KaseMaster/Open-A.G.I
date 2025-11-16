"""
CertifiedMath API Models for Open-A.G.I Integration
Pydantic models for CertifiedMath operations
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class CertifiedMathOperationRequest(BaseModel):
    """Request model for CertifiedMath operations"""
    operand_a: str = Field(..., description="First operand as string")
    operand_b: Optional[str] = Field(None, description="Second operand as string (for binary operations)")
    iterations: Optional[int] = Field(20, description="Number of iterations for sqrt and phi_series operations")
    pqc_cid: Optional[str] = Field(None, description="PQC commit ID for attestation")
    
class CertifiedMathOperationResponse(BaseModel):
    """Response model for CertifiedMath operations"""
    success: bool = True
    result: Optional[str] = None
    log_hash: Optional[str] = None
    pqc_cid: Optional[str] = None
    operation: str
    iterations: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class CertifiedMathExportRequest(BaseModel):
    """Request model for exporting audit logs"""
    path: str = Field(..., description="Path to export the audit log to")
    
class CertifiedMathExportResponse(BaseModel):
    """Response model for exporting audit logs"""
    success: bool = True
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)