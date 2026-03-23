import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dl
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import h5py
import numpy as np
from tqdm import tqdm

from models.unet import UNet2, UNet, UNetBilinear
data = {}

def pad_to_512(x):
    """
    x: (B, C, 500, 500)
    returns: (B, C, 512, 512)
    """
    pad = (6, 6, 6, 6)  # (left, right, top, bottom)
    return F.pad(x, pad, mode="constant", value=0.0)

def crop_to_500(x):
    """
    x: (B, C, 512, 512)
    returns: (B, C, 500, 500)
    """
    return x[:, :, 6:-6, 6:-6]

def parse_tree(obj, indent=0, current=None):
    if current is None:
        current = data

    for key in obj:
        item = obj[key]
        if isinstance(item, h5py.Group):
            current[key] = {}
            parse_tree(item, indent + 1, current=current[key])
        else:
            # store dataset as a numpy array in the current group's dict
            current[key] = np.asarray(item[()])

class GridDataset(Dataset):
    def __init__(self, grids, h, dhdx, dhdy):
        self.grids = grids
        self.h = h
        self.dhdx = dhdx
        self.dhdy = dhdy

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self.h[idx],
            self.dhdx[idx],
            self.dhdy[idx]
        )

def gradients(h):
    dhdx = h[:, :, :, 1:] - h[:, :, :, :-1]
    dhdy = h[:, :, 1:, :] - h[:, :, :-1, :]
    return dhdx, dhdy

def boundary_loss(pred, true):
    return (
        (pred[:, :, 0, :] - true[:, :, 0, :]).abs().mean() +
        (pred[:, :, -1, :] - true[:, :, -1, :]).abs().mean() +
        (pred[:, :, :, 0] - true[:, :, :, 0]).abs().mean() +
        (pred[:, :, :, -1] - true[:, :, :, -1]).abs().mean()
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetBilinear(in_channels=1, out_channels=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#load data from h5 file
with h5py.File("data/grids_data_512x512.h5", "r") as f:
    parse_tree(f)


# Prepare data
grids = data["grid"]
h     = data["h"]
dhdx  = data["dhdx"]
dhdy  = data["dhdy"]

keys = sorted(grids.keys())

dhdx_arr = np.stack([dhdx[k] for k in keys])
dhdy_arr = np.stack([dhdy[k] for k in keys])

dhdx_arr = dhdx_arr[:, None, :, :]  # (N, 1, H, W)
dhdy_arr = dhdy_arr[:, None, :, :]

grids_x = np.stack([grids[k] for k in keys])
Y = np.stack([h[k] for k in keys])

grids_x = grids_x[:, None, :, :]  # (N, 1, 512, 512)
Y = Y[:, None, :, :]
grids = torch.from_numpy(grids_x).float()
h     = torch.from_numpy(Y).float()

## Train-test split

grids_train, grids_test, \
h_train, h_test, \
dhdx_train, dhdx_test, \
dhdy_train, dhdy_test = train_test_split(
    grids, h, dhdx_arr, dhdy_arr,
    test_size=0.2,
    random_state=42
)

dx = 1.0 / (grids.shape[-1] - 1)
dhdx_train *= dx
dhdx_test  *= dx
dhdy_train *= dx
dhdy_test  *= dx
# normalize h only
h_mean = h.mean()
h_std  = h.std() + 1e-8
h_train = (h_train - h_mean) / h_std
h_test  = (h_test  - h_mean) / h_std

dhdx_train = dhdx_train / h_std
dhdx_test  = dhdx_test  / h_std

dhdy_train = dhdy_train / h_std
dhdy_test  = dhdy_test  / h_std

train_dataset = GridDataset(grids_train, h_train, dhdx_train, dhdy_train)
test_dataset  = GridDataset(grids_test,  h_test,  dhdx_test,  dhdy_test)

train_loader = dl(
    train_dataset,
    batch_size=2,
    shuffle=True
)

test_loader = dl(
    test_dataset,
    batch_size=2,
    shuffle=False
)
print(train_loader.__len__())
print(test_loader.__len__())

epochs = 40

print("h:", h_train.min().item(), h_train.max().item())
print("dhdx:", dhdx_train.min().item(), dhdx_train.max().item())
print("dhdy:", dhdy_train.min().item(), dhdy_train.max().item())

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for grid, h_true, dhdx_true, dhdy_true in pbar:

        grid = grid.to(device)
        h_true = h_true.to(device)
        dhdx_true = dhdx_true.to(device)
        dhdy_true = dhdy_true.to(device)

        optimizer.zero_grad()

        h_pred = model(grid)

        mse = criterion(h_pred, h_true)
        dhdx_pred, dhdy_pred = gradients(h_pred)
        loss_grad = (
            F.mse_loss(dhdx_pred, dhdx_true[:, :, :, :-1]) +
            F.mse_loss(dhdy_pred, dhdy_true[:, :, :-1, :])
        )
        loss_bc = boundary_loss(h_pred, h_true)
        loss = mse + 0.5 * loss_grad + 0.1 * loss_bc
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.6f}")

model.eval()   # disables dropout, batchnorm
test_loss = 0.0

pbar = tqdm(test_loader, desc="Testing", leave=False)
i = 0
with torch.no_grad():
    for grid, h_true, dhdx_true, dhdy_true in pbar:
        
        grid = grid.to(device)
        h_true = h_true.to(device)

        h_pred = model(grid)

        loss = criterion(h_pred, h_true)
        test_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # save with matplotlib image of test results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(h_pred.cpu().numpy()[0, 0], cmap='viridis')
        plt.title("Predicted")
        plt.subplot(1, 2, 2)
        plt.imshow(h_true.cpu().numpy()[0, 0], cmap='viridis')
        plt.title("True")
        plt.savefig(f"results/test_result_{epoch}_{i}.png")
        i += 1

test_loss /= len(test_loader)
print("Test MSE:", test_loss)

torch.save(model.state_dict(), "weights/grads_loss_unetbilinear_model_e2.pth")