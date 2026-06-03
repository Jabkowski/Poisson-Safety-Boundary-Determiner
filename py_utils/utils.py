from torch.utils.data import DataLoader as dl
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import torch.nn.functional as F


def gradients(h):
    dhdx = h[:, :, :, 1:] - h[:, :, :, :-1]
    dhdy = h[:, :, 1:, :] - h[:, :, :-1, :]
    return dhdx, dhdy


def calc_grad_loss(criterion, h_pred, h_true):
    mse = criterion(h_pred, h_true)
    # --- gradients (computed consistently!) ---
    dhdx_pred, dhdy_pred = gradients(h_pred)
    dhdx_true, dhdy_true = gradients(h_true)

    loss_grad = F.mse_loss(dhdx_pred, dhdx_true) + F.mse_loss(dhdy_pred, dhdy_true)
    # --- boundary ---
    loss_bc = boundary_loss(h_pred, h_true)
    # # --- smoothness ---
    # loss_smooth = dhdx_pred.abs().mean() + dhdy_pred.abs().mean()

    ## multiply gradd loss by a factor to balance with mse
    loss_grad = 30.0 * loss_grad

    return mse, loss_grad, loss_bc


def load_grid_h5_to_torch(h5_path):
    with h5py.File(h5_path, "r") as f:
        data = parse_tree(f)
        grids = data["grid"]
        keys = sorted(grids.keys())
        grids_x = np.stack([grids[k] for k in keys])
        grids_x = grids_x[:, None, :, :]  # (N, 1, 512, 512)
        grids = torch.from_numpy(grids_x).float()
        return grids


def load_h5_to_torch(h5_path):
    with h5py.File(h5_path, "r") as f:
        data = parse_tree(f)

    grids = data["grid"]
    h = data["h"]

    keys = sorted(grids.keys())

    grids_x = np.stack([grids[k] for k in keys])
    Y = np.stack([h[k] for k in keys])

    grids_x = grids_x[:, None, :, :]  # (N, 1, 512, 512)
    Y = Y[:, None, :, :]
    grids = torch.from_numpy(grids_x).float()
    h = torch.from_numpy(Y).float()

    return grids, h


def plot_inference_results(h_pred, grid, i):
    h_pred = h_pred.detach().cpu().numpy()
    grid = grid.detach().cpu().numpy()
    grid_map = grid[0, 0]
    h_pred_map = h_pred[0, 0]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title(f"Predicted h")
    im_pred = axes[0].imshow(h_pred_map, cmap="viridis")
    fig.colorbar(im_pred, ax=axes[0])

    axes[1].set_title(f"Grid")
    im_true = axes[1].imshow(grid_map, cmap="viridis")
    fig.colorbar(im_true, ax=axes[1])
    fig.tight_layout()
    fig.savefig(f"results/infer_carla_grid_{i}.png")
    plt.close(fig)


def plot_validation_results(h_pred, h_true, epoch_label="", save_plot=False, i=0):
    h_pred = h_pred.detach().cpu().numpy()
    h_true = h_true.detach().cpu().numpy()

    # Keep both plots on the same color scale using the true field range.
    true_map = h_true[0, 0]
    pred_map = h_pred[0, 0]
    vmin = float(np.nanmin(true_map))
    vmax = float(np.nanmax(true_map))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    # Plotting
    import matplotlib.pyplot as plt
    from matplotlib import colors

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title(f"Predicted h (epoch {epoch_label})")
    im_pred = axes[0].imshow(pred_map, cmap="viridis", norm=norm)
    fig.colorbar(im_pred, ax=axes[0])

    axes[1].set_title(f"True h (epoch {epoch_label})")
    im_true = axes[1].imshow(true_map, cmap="viridis", norm=norm)
    fig.colorbar(im_true, ax=axes[1])

    fig.tight_layout()
    if save_plot:
        fig.savefig(f"results/test_result_{epoch_label}_{i}.png")
        plt.close(fig)
    else:
        plt.show()


def parse_tree(obj, indent=0, current=None):
    data = {}
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

    return data


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


class GridDataset(Dataset):
    def __init__(self, grids, h):
        self.grids = grids
        self.h = h

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return self.grids[idx], self.h[idx]


def boundary_loss(pred, true):
    return (
        (pred[:, :, 0, :] - true[:, :, 0, :]).abs().mean()
        + (pred[:, :, -1, :] - true[:, :, -1, :]).abs().mean()
        + (pred[:, :, :, 0] - true[:, :, :, 0]).abs().mean()
        + (pred[:, :, :, -1] - true[:, :, :, -1]).abs().mean()
    )
