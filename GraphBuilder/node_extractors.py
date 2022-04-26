import torch
import numpy as np
from abc import ABC, abstractmethod
from .constants import MAX_SAMPLES, NUM_CHANNELS


class BaseNodeExtractor(ABC):
    
    @abstractmethod
    def __init__(self) -> None: ...
    
    @abstractmethod
    def extract_features(self, data: np.ndarray): ...
    
    
class RawExtractor(BaseNodeExtractor):
    
    def __init__(self) -> None:
        super().__init__()
        self.MAX_SAMPLES = MAX_SAMPLES
        
        
    def extract_features(self, data):
        #node_features = np.pad(data, pad_width=((0, 0), (0,  MAX_SAMPLES - data.shape[1])), constant_values=(0, ))
        node_features = torch.from_numpy(data)
        return node_features