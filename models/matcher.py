import numpy as np
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BioMatchingEngine")

class SequenceMatchingEngine:
    """
    Production-grade mathematical alignment engine executing high-dimensional 
    matrix operations for complex biological profiling.
    """
    def __init__(self, registry_matrix: np.ndarray, weight_vector: np.ndarray = None):
        if registry_matrix.ndim != 2:
            raise ValueError("Registry matrix must be exactly 2-dimensional.")
        
        self.registry = registry_matrix
        self.num_donors, self.dimensions = registry_matrix.shape
        
        if weight_vector is not None:
            if weight_vector.shape != (self.dimensions,):
                raise ValueError("Weight vector dimensions must match registry dimensions.")
            self.weights = weight_vector
        else:
            self.weights = np.ones(self.dimensions)
        
        logger.info(f"Initialized Engine with {self.num_donors} profiles across {self.dimensions} dimensions.")

    def _calculate_weighted_cosine(self, vector_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        w_a = vector_a * self.weights
        w_b = matrix_b * self.weights
        
        dot_products = np.dot(w_b, w_a)
        norm_a = np.linalg.norm(w_a)
        norm_b = np.linalg.norm(w_b, axis=1)
        
        denominator = norm_a * norm_b
        denominator[denominator == 0] = 1e-9
        
        return dot_products / denominator

    def execute_alignment_pipeline(self, patient_profile: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        if patient_profile.shape != (self.dimensions,):
            logger.error("Input profile dimensionality mismatch error.")
            raise ValueError(f"Expected vector size {self.dimensions}, got {patient_profile.shape}.")
            
        logger.info("Executing vectorized alignment matrix operations...")
        similarities = self._calculate_weighted_cosine(patient_profile, self.registry)
        top_indexes = np.argsort(similarities)[::-1][:top_n]
        
        output_payload = []
        for index in top_indexes:
            score = float(similarities[index])
            tier = "Tier-1 (Optimal)" if score >= 0.85 else "Tier-2 (Moderate)"
            
            output_payload.append({
                "donor_index_id": int(index),
                "alignment_score": round(score, 6),
                "matching_tier": tier
            })
            
        logger.info(f"Successfully calculated and extracted top {top_n} biological sequences.")
        return output_payload