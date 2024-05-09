import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn
from tqdm import tqdm
from einops import rearrange
from sdf.utils import trilinear_interpolation
from sdf.config import hash_param





#Hyperparameters
#***********************************************
min_grid_res = hash_param.get("min_grid_res")
max_grid_res = hash_param.get("max_grid_res")
num_LOD = hash_param.get("num_LOD")
band_width = hash_param.get("band_width")
#***********************************

print(min_grid_res)


class HashGrid(nn.Module):
    def __init__(self, min_grid_res=min_grid_res, max_grid_res=max_grid_res, num_LOD=num_LOD, band_width=band_width):
        super().__init__()
        self.feat_dim = 1 # feature dim size
        self.codebook = nn.ParameterList([])
        self.codebook_size = 2**band_width

        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_LOD - 1))
        self.LODS = [int(1 + np.floor(min_grid_res * (b**l))) for l in range(num_LOD)]
        print("LODS:", self.LODS)
        self.init_hash_structure()

    def init_hash_structure(self):
        for LOD in self.LODS:
            num_pts = LOD**3
            fts = torch.zeros(min(self.codebook_size, num_pts), self.feat_dim)
            fts += torch.randn_like(fts) * 0.01
            fts = nn.Parameter(fts)
            self.codebook.append(fts)

    def forward(self, pts):
        _, feat_dim = self.codebook[0].shape
        pts = pts / 2 + 0.5
        feats = []
        for i, res in enumerate(self.LODS):

            features = trilinear_interpolation(res, self.codebook[i], pts, "HASH")

            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.sum(-1)
