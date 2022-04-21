""" EEGDataset is a class used to handle the EEG data
and build the corresponding graphs from it.
"""

import torch
import numpy as np
import pandas as pd
import math
from torch_geometric.data import Data, Dataset
from itertools import product


class EEGDataset(Dataset):

    def __init__(self, X, y, indices, loader_type, sfreq, transform=None):
        self.epochs = X
        self.labels = y
        self.indices = indices
        self.loader_type = loader_type
        self.sfreq = sfreq
        self.transform = transform

        self.ch_names = [
            "F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3",
            "O2-P4"
        ]

        self.ref_names = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]

        self.nodes_ids = range(len(self.ch_names))
        self.edge_index = torch.tensor(
            [[a, b] for a, b in product(self.nodes_ids, self.nodes_ids)]).t().contiguous()

        #self.distances = self.get_sensor_distances()
        #dist = np.array(self.distances)
        
        self.spec_coh_values = np.load(
            "./data/numpy_arrays/X_spec_coh_epilepsy_corpus.npy", 
            allow_pickle=True
        )
        
        self.distances = np.zeros_like(self.spec_coh_values)#(dist - np.min(dist)) / (np.max(dist) - np.min(dist))

    def get_sensor_distance(self):
        coords_1010 = pd.read_csv("./data/docs/standard_1010.tsv.txt",
                                  sep='\t')
        num_edges = self.edge_index.shape[1]
        distances = []

        for idx in range(num_edges):
            sensor1_idx = self.edge_index[0, idx]
            sensor2_idx = self.edge_index[1, idx]
            dist = self.get_geodesic_distance(sensor1_idx, sensor2_idx,
                                              coords_1010)
            distances.append(dist)

        assert len(distances) == num_edges
        return distances

    def get_geodesic_distance(self, montage_sensor1_idx, montage_sensor2_idx,
                              coords_1010):

        # get the reference sensor in the 10-10 system for the current montage pair in 10-20 system
        ref_sensor1 = self.ref_names[montage_sensor1_idx]
        ref_sensor2 = self.ref_names[montage_sensor2_idx]

        x1 = float(coords_1010[coords_1010.label == ref_sensor1]["x"])
        y1 = float(coords_1010[coords_1010.label == ref_sensor1]["y"])
        z1 = float(coords_1010[coords_1010.label == ref_sensor1]["z"])

        # print(ref_sensor2, montage_sensor2_idx, coords_1010[coords_1010.label == ref_sensor2]["x"])
        x2 = float(coords_1010[coords_1010.label == ref_sensor2]["x"])
        y2 = float(coords_1010[coords_1010.label == ref_sensor2]["y"])
        z2 = float(coords_1010[coords_1010.label == ref_sensor2]["z"])

        r = 1  # since coords are on unit sphere
        # rounding is for numerical stability, domain is [-1, 1]
        dist = r * math.acos(
            round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2))
        return dist

    def __len__(self):
        return len(self.indices)
    
    def _get_labels(self, label):
        if label == "healty":
            return torch.tensor([[0, 1]], dtype=torch.float64)
        return torch.tensor([[1, 0]], dtype=torch.float64)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # map input idx (ranging from 0 to __len__() inside self.indices) to an idx in the whole dataset (inside self.epochs)
        # assert idx < len(self.indices)
        idx = self.indices[idx]

        node_features = self.epochs[idx]
        node_features = torch.from_numpy(node_features.reshape(8, 6))

        # spectral coherence between 2 montage channels!
        spec_coh_values = self.spec_coh_values[idx, :]
        # combine edge weights and spect coh values into one value/ one E x 1 tensor
        edge_weights = self.distances + spec_coh_values
        edge_weights = torch.tensor(edge_weights)

       
        data = Data(x=node_features,
                    edge_index=self.edge_index,
                    edge_attr=edge_weights,
                    dataset_idx=idx,
                    y=self._get_labels(self.labels[idx])
                    # pos=None, norm=None, face=None, **kwargs
                    )
        return data