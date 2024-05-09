import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn




class MLP(nn.Module):
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=3
    ):
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Linear(input_dim, hidden_dim, bias=True))
        for i in range(num_hidden_layers):
            osize = hidden_dim if i < num_hidden_layers - 1 else output_dim
            self.module_list.append(torch.nn.ReLU())
            self.module_list.append(torch.nn.Linear(hidden_dim, osize, bias=True))
        self.fc1 = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure

    def forward(self, coords):

        
        feat = self.grid_structure(coords)
        out = (self.fc1(feat))
        return out