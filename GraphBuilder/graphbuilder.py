import imp
import torch
from itertools import product
from abc import ABC, abstractmethod
from cProfile import label
from torch_geometric.data import Data
from node_extractors import RawExtractor
from edge_extractors import PearsonExtractor
from constants import NUM_CHANNELS


class BaseGraphBuilder(ABC):
    
    @abstractmethod
    def __init__(self) -> None: 
        self.node_feature_extractor = None
        self.edge_feature_extractor = None
        
    
    @abstractmethod
    def build(self, data, label):
        node_features = self.node_feature_extractor.extract_features(data)
        edge_features = self.edge_feature_extractor.extract_features(data)
        format_label = self.format_label()
        edge_index = torch.tensor(
            [[a, b] for a, b in product(range(NUM_CHANNELS), range(NUM_CHANNELS))]
        ).t().contiguous()
        
        graph = Data(
            x=node_features,
            edge_attr=edge_features,
            label=format_label,
            edge_index=edge_index
        )
        
        return graph
        
    def format_label(self) -> torch.Tensor:
        return torch.tensor([0, 1, 0], dtype=torch.float64)
        
            
class RawAndPearson(BaseGraphBuilder):
    def __init__(self) -> None:
        self.node_feature_extractor = RawExtractor()
        self.edge_feature_extractor = PearsonExtractor()
        
        
    def build(self, data, label):
        return super().build(data, label)
        
    
    
