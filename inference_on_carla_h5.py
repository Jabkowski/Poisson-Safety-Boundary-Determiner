import os
import torch

from models.unet import UNetBilinear
from py_utils.utils import (
    load_grid_h5_to_torch,
    plot_inference_results,
)

h5_path = "carla_generator/grids/carla_grid.h5"  #  carla_generator/grids/carla_grid.h5 , data/grids_data_512x512.h5
WEIGHTS_PATH = "weights/grads_loss_v3_unetbilinear_model_e15.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetBilinear(in_channels=1, out_channels=1).to(device)

# load data from carla h5 file
grids = load_grid_h5_to_torch(h5_path)

if os.path.isfile(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        for i in range(grids.shape[0]):
            grid = grids[i : i + 1].to(device)  # (1, 1, 512, 512)
            h_pred = model(grid)  # (1, 1, 512, 512)
            print(f"Predicted h for grid {i}:")
            plot_inference_results(h_pred, grid, i)
