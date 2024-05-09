import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn
from tqdm import tqdm
from einops import rearrange
from scipy.spatial import KDTree
from sdf.utils import trilinear_interpolation
from sdf.config import dense_param


#Hyperparameters
#***********************************************
base_lod = dense_param.get("base_lod")
num_lod = dense_param.get("num_lod")
grid_dim = dense_param.get("grid_dim")
interpolation_type = dense_param.get("interpolation_type")
#***********************************


class DenseGrid(nn.Module):
    def __init__(self, base_lod =base_lod, num_lod = num_lod, grid_dim = grid_dim , interpolation_type=interpolation_type):
        super().__init__()
        self.feat_dim = 1# feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # TODO implement trilinear interpolation
        self.grid_dim = grid_dim
        self.LODS = [2**L for L in range(base_lod, base_lod + num_lod)]

        print("LODS:", self.LODS)
        self.init_feature_structure()


    def init_feature_structure(self):
        for LOD in self.LODS:
            fts = torch.zeros(LOD**self.grid_dim, self.feat_dim)
            fts += torch.randn_like(fts) * 0.01
            fts = nn.Parameter(fts)
            self.codebook.append(fts)
        
    

    def forward(self, pts):
        feats = []
        pts = pts / 2 + 0.5 # normalzing the input .. for better performance 
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            if self.interpolation_type == "closest":
                x = pts[:, 0] * (res - 1)
                x = torch.floor(x).int()
                y = pts[:, 1] * (res - 1)
                y = torch.floor(y).int()
                z = pts[:, 2] * (res - 1)
                z = torch.floor(z).int()
                indices = (x + y * res + z * res * res).long()
                features = self.codebook[i][indices]
            elif self.interpolation_type == "trilinear":
                features = trilinear_interpolation(res, self.codebook[i], pts, "NGLOD")
            
            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)