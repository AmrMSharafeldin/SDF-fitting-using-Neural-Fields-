from sdf.train import train_model
from sdf.hashgrid import HashGrid
from sdf.dense_grid import DenseGrid
from sdf.mlp import MLP
import torch

def create_model(model_type):
        if(model_type == "hash"):
            smart_grid = HashGrid()
        elif(model_type == "dense"):
            smart_grid = DenseGrid()
        return MLP(smart_grid, 1, 64, 1).to(device='cuda')


class SDF_NN():
    def __init__(self, coords, values , model_type):
        self.model = create_model(model_type)
        train_model(coords , values , self.model , model_type = model_type)
    def __call__(self, x):
        with torch.no_grad():
            coords_tensor = torch.tensor(x).to(torch.float32).to(device='cuda')
            out = self.model(coords_tensor)
            return (torch.squeeze(out).cpu()).numpy()
        