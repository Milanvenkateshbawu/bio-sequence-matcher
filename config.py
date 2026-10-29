import numpy as np

class SystemConfig:
    """Central production environment attributes for the ML pipeline."""
    VECTOR_DIMENSIONS: int = 128
    REGISTRY_SIZE: int = 25000  
    DEFAULT_MATCH_COUNT: int = 5
    
    @classmethod
    def generate_global_weights(cls) -> np.ndarray:
        np.random.seed(42) 
        raw_weights = np.random.uniform(0.5, 2.0, cls.VECTOR_DIMENSIONS)
        return raw_weights / np.sum(raw_weights)