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
    def __init__(self, grids, h):
        """
        grids: Tensor (N, 1, H, W)
        h:     Tensor (N, 1, H, W)
        """
        self.grids = grids
        self.h = h

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return self.grids[idx], self.h[idx]

def gradient_loss(h):
    dx = h[:, :, :, 1:] - h[:, :, :, :-1]
    dy = h[:, :, 1:, :] - h[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()

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
with h5py.File("data/grids_data.h5", "r") as f:
    parse_tree(f)


# Prepare data
grids = data["grid"]
h     = data["h"]

keys = sorted(grids.keys())

grids_x = np.stack([grids[k] for k in keys])
Y = np.stack([h[k] for k in keys])

grids_x = grids_x[:, None, :, :]  # (N, 1, 500, 500)
Y = Y[:, None, :, :]
grids = torch.from_numpy(grids_x).float()
h     = torch.from_numpy(Y).float()

# normalize h only
h_mean = h.mean()
h_std  = h.std() + 1e-8
h = (h - h_mean) / h_std

## Train-test split

grids_train, grids_test, h_train, h_test = train_test_split(
    grids, h, test_size=0.2, random_state=42
)

train_dataset = GridDataset(grids_train, h_train)
test_dataset  = GridDataset(grids_test,  h_test)

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

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for grid, h_true in pbar:
        
        grid = pad_to_512(grid)
        h_true = pad_to_512(h_true)

        grid = grid.to(device)
        h_true = h_true.to(device)

        optimizer.zero_grad()

        h_pred = model(grid)
        h_pred = crop_to_500(h_pred)
        h_true = crop_to_500(h_true)

        mse = criterion(h_pred, h_true)
        smooth = gradient_loss(h_pred)

        loss = mse + 0.05 * smooth + 0.1 * boundary_loss(h_pred, h_true)
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
    for grid, h_true in pbar:

        grid = pad_to_512(grid)
        h_true = pad_to_512(h_true)
        
        grid = grid.to(device)
        h_true = h_true.to(device)

        h_pred = model(grid)

        h_pred = crop_to_500(h_pred)
        h_true = crop_to_500(h_true)

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

torch.save(model.state_dict(), "weights/unetbilinear_model.pth")