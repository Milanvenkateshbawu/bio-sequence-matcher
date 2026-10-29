import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import numpy as np

from config import SystemConfig
from data_pipeline.generator import BiologicalDataGenerator
from models.matcher import SequenceMatchingEngine

app = FastAPI(
    title="Bio-Sequence Intelligent Matching Pipeline",
    description="Enterprise-grade architecture utilizing high-dimensional matrix vectorization.",
    version="2.0.0"
)

try:
    config = SystemConfig()
    data_gen = BiologicalDataGenerator(dimensions=config.VECTOR_DIMENSIONS)
    global_registry = data_gen.generate_registry_database(num_donors=config.REGISTRY_SIZE)
    system_weights = config.generate_global_weights()
    matching_engine = SequenceMatchingEngine(global_registry, weight_vector=system_weights)
except Exception as init_error:
    raise RuntimeError(f"System initialization failure: {str(init_error)}")

class AlignmentResponse(BaseModel):
    donor_index_id: int = Field(..., description="The index locator inside the system matrix.")
    alignment_score: float = Field(..., description="Calculated cosine similarity vector metric.")
    matching_tier: str = Field(..., description="Classification tier based on strict statistical bounds.")

@app.post(
    "/api/v2/align", 
    response_model=List[AlignmentResponse],
    status_code=status.HTTP_200_OK,
    summary="Execute multi-tier vector alignment matching"
)
def run_pipeline():
    try:
        simulated_patient = data_gen.generate_patient_profile()
        match_results = matching_engine.execute_alignment_pipeline(
            simulated_patient, 
            top_n=config.DEFAULT_MATCH_COUNT
        )
        return match_results
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as sys_err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(sys_err))

@app.get("/health", summary="Perform MLOps system diagnostics check")
def health_check():
    return {"status": "healthy", "monitored_registry_depth": config.REGISTRY_SIZE}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)