import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_add_pool


class EEGGNN(nn.Module):
    
    def __init__(self, reduced_sensors, sfreq=None, batch_size=32):
        super(EEGGNN, self).__init__()
        # Define and initialize hyperparameters
        self.sfreq = sfreq
        self.batch_size = batch_size
        self.input_size = 8 if reduced_sensors else 62
        
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(6, 16, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm = BatchNorm(50)
        
        # Fully connected layers
        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weigth))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weigth))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weigth))
        conv_out = F.leaky_relu(self.conv4(x, edge_index, edge_weigth))
        
        # Perform batch normalization
        batch_norm_out = F.leaky_relu(conv_out)
        
        # Global add pooling
        mean_pool = global_add_pool(batch_norm_out, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        out = F.leaky_relu(self.fc3(out))
        return F.softmax(out, dim=1)
        
        