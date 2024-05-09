import numpy as np
from utils import plot_points, download_data , load_data , viz
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sdf.utils import MAPE,viz






def train_model( coords, values , model ,
                learning_rate = 1.0e-3  , max_epochs = 100,loss = "MSE" , model_type = "hash"):
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coords_tensor = torch.tensor(coords).to(torch.float32).to(device)
    values_tensor = torch.tensor(values).to(torch.float32).to(device)
    values_tensor =torch.unsqueeze(values_tensor , -1)

    loss_fn = torch.nn.MSELoss() 
    if (model_type == "hash"):
        model.grid_structure.weight_decay = 0.0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    loop = tqdm(range(max_epochs))

    for epoch in loop:
        coords, values = coords_tensor, values_tensor

        output = model(coords)
        loss = loss_fn(output, values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch: {epoch}")
        loop.set_postfix_str(
            f" Loss: {loss.item():.5f}"
        )
        #if epoch % 10 == 0:
            #viz(coords, output , epoch)