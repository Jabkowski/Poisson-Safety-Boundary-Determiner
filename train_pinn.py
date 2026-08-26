import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import copy
from scipy.ndimage import binary_erosion, label, center_of_mass
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from models.unet import UNet, UNetBilinear

# Ustawienie ziarna losowości dla powtarzalności wyników
torch.manual_seed(42)
np.random.seed(42)

class DoubleConv(nn.Module):
    """(splot2d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSubNetwork(nn.Module):
    """Dedykowana architektura UNet dla pojedynczego zadania fizycznego"""

    def __init__(self, in_channels=1, out_channels=1):
        super(UNetSubNetwork, self).__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(64, 128)
        )

        self.up1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv_up1 = DoubleConv(128 + 64, 64)

        self.up2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv_up2 = DoubleConv(64 + 32, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)

        return self.outc(x)


class UNetDoublePoisson(nn.Module):
    """
    Splotowy model U-Net realizujący strategię dwóch niezależnych sieci:
    - u_net: generuje pole wektorowe u [B, 2, H, W]
    - h_net: generuje pole bezpieczeństwa h [B, 1, H, W]
    Wejście: siatka zajętości [B, 1, H, W]
    Wyjście: połączony rozkład [B, 3, H, W] (u_x, u_y, h)
    """

    def __init__(self):
        super(UNetDoublePoisson, self).__init__()
        self.u_net = UNetBilinear(in_channels=1, out_channels=2)
        self.h_net = UNetBilinear(in_channels=2, out_channels=1)
        # self.u_net = UNetSubNetwork(in_channels=1, out_channels=2)
        # self.h_net = UNetSubNetwork(in_channels=2, out_channels=1)
        # self.h_net = UNet(in_channels=2, out_channels=1)

    def forward(self, x):
        u = self.u_net(x)
        h = self.h_net(u)
        return torch.cat([u, h], dim=1)


class H5PoissonDataset(Dataset):
    """
    Klasa Dataset wczytująca wszystkie wygenerowane mapy i powiązane rozkłady h
    z pliku HDF5 bezpośrednio do pamięci RAM, dla maksymalnej wydajności i stabilności na GPU.
    """

    def __init__(self, file_path):
        self.grids = []
        self.h_values = []
        self.dhdx_values = []
        self.dhdy_values = []
        self.ux_values = []
        self.uy_values = []
        self.normalization_stats = None

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Nie znaleziono pliku bazy danych: {file_path}"
            )

        print(f"Ładowanie zestawu danych z pliku: {file_path}...")
        with h5py.File(file_path, "r") as f:
            # Sortujemy klucze (indeksy map) dla zachowania spójności
            keys = sorted(list(f["grid"].keys()))
            for key in tqdm(keys, desc="Wczytywanie map"):
                # Obrazy w bazie są zapisane jako [H, W] lub [1, H, W];
                # normalizujemy do [1, H, W], bo model oczekuje wejścia 4D [B, C, H, W].
                def _as_channel_tensor(data):
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    return torch.tensor(arr, dtype=torch.float32)

                grid_data = _as_channel_tensor(f["grid"][key][:])
                h_data = _as_channel_tensor(f["h"][key][:])
                dhdx_data = _as_channel_tensor(f["dhdx"][key][:])
                dhdy_data = _as_channel_tensor(f["dhdy"][key][:])
                ux_data = _as_channel_tensor(f["u_x"][key][:])
                uy_data = _as_channel_tensor(f["u_y"][key][:])

                self.grids.append(grid_data)
                self.h_values.append(h_data)
                self.dhdx_values.append(dhdx_data)
                self.dhdy_values.append(dhdy_data)
                self.ux_values.append(ux_data)
                self.uy_values.append(uy_data)

    def set_normalization_stats(self, normalization_stats):
        self.normalization_stats = normalization_stats

    def _normalize_field(self, field_name, tensor):
        if self.normalization_stats is None:
            return tensor
        if field_name not in self.normalization_stats:
            return tensor
        mean = self.normalization_stats[field_name]["mean"]
        std = self.normalization_stats[field_name]["std"]
        return (tensor - mean) / std

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self._normalize_field("h", self.h_values[idx]),
            self._normalize_field("dhdx", self.dhdx_values[idx]),
            self._normalize_field("dhdy", self.dhdy_values[idx]),
            self._normalize_field("ux", self.ux_values[idx]),
            self._normalize_field("uy", self.uy_values[idx]),
        )


def compute_normalization_stats(dataset, train_indices, eps=1e-8):
    """
    Wylicza statystyki normalizacji wyłącznie na podzbiorze treningowym.
    Statystyki mają postać: {field: {mean: float, std: float}}.
    """
    field_mapping = {
        "h": "h_values",
        "dhdx": "dhdx_values",
        "dhdy": "dhdy_values",
        "ux": "ux_values",
        "uy": "uy_values",
    }

    stats = {}
    for field_name, attr_name in field_mapping.items():
        values = getattr(dataset, attr_name)
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0

        for idx in train_indices:
            tensor = values[idx].to(dtype=torch.float64)
            total_sum += tensor.sum().item()
            total_sum_sq += (tensor * tensor).sum().item()
            total_count += tensor.numel()

        mean = total_sum / max(total_count, 1)
        variance = max(total_sum_sq / max(total_count, 1) - mean * mean, 0.0)
        std = float(np.sqrt(variance) + eps)

        stats[field_name] = {"mean": float(mean), "std": std}

    return stats


def denormalize_pred_3ch(pred_3ch, normalization_stats):
    """
    Konwertuje wyjście modelu [ux_norm, uy_norm, h_norm] do skali fizycznej.
    """
    pred_phys = pred_3ch.clone()

    ux_mean = normalization_stats["ux"]["mean"]
    ux_std = normalization_stats["ux"]["std"]
    uy_mean = normalization_stats["uy"]["mean"]
    uy_std = normalization_stats["uy"]["std"]
    h_mean = normalization_stats["h"]["mean"]
    h_std = normalization_stats["h"]["std"]

    pred_phys[:, 0:1, :, :] = pred_phys[:, 0:1, :, :] * ux_std + ux_mean
    pred_phys[:, 1:2, :, :] = pred_phys[:, 1:2, :, :] * uy_std + uy_mean
    pred_phys[:, 2:3, :, :] = pred_phys[:, 2:3, :, :] * h_std + h_mean
    return pred_phys


def compute_batch_boundary_u_targets(grid_batch, dx=10.0 / 128.0):
    """
    Dla każdej siatki zajętości w pacce (Batch) wykrywa spójne przeszkody,
    oblicza ich środki ciężkości i tworzy docelowy tensor u_target na krawędziach przeszkód.
    """
    device = grid_batch.device
    B, _, H, W = grid_batch.shape
    U_targets = torch.zeros(
        (B, 2, H, W), dtype=torch.float32, device=device
    )
    boundary_masks = torch.zeros(
        (B, 1, H, W), dtype=torch.float32, device=device
    )

    grid_cpu = grid_batch.squeeze(1).detach().cpu().numpy()

    for b in range(B):
        grid = grid_cpu[b]

        # Ekstrakcja krawędzi przeszkody za pomocą różnicy dylatacji/erozji
        eroded = -F.max_pool2d(
            -grid_batch[b : b + 1].float(),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        boundary_tensor = grid_batch[b : b + 1].float() - eroded
        boundary_masks[b : b + 1] = boundary_tensor

        boundary_np = boundary_tensor.squeeze().cpu().numpy() > 0.5
        if not np.any(boundary_np):
            continue

        # Segmentacja przeszkód
        labeled, num_features = label(grid)
        centers = {}
        for i in range(1, num_features + 1):
            row_c, col_c = center_of_mass(labeled == i)
            # Mapowanie indeksu siatki na współrzędne świata [-5.0, 5.0]
            cx = -5.0 + (col_c / (W - 1)) * 10.0
            cy = -5.0 + (row_c / (H - 1)) * 10.0
            centers[i] = (cx, cy)

    return U_targets, boundary_masks


def compute_boundary_mask(grid_batch):
    """
    Zwraca maskę krawędzi przeszkód [B, 1, H, W] używaną do warunku brzegowego dla u.
    """
    boundary_masks = torch.zeros_like(grid_batch, dtype=torch.float32)
    for b in range(grid_batch.shape[0]):
        eroded = -F.max_pool2d(
            -grid_batch[b : b + 1].float(),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        boundary_tensor = grid_batch[b : b + 1].float() - eroded
        boundary_masks[b : b + 1] = boundary_tensor
    return boundary_masks


def calc_poisson_pinn_loss(
    pred_3ch,
    h_true,
    dhdx_true,
    dhdy_true,
    ux_true,
    uy_true,
    grid,
    dx=10.0 / 128.0,
    detach_u=True,
):
    """
    Oblicza fizyczny błąd PDE (Physics-Informed Loss) przy użyciu splotów 2D dla paczki (Batch).
    detach_u=True blokuje przepływ wsteczny gradientów h do podsieci pola pomocniczego u.
    Używa skończonych różnic (filtry Sobela, Laplacjan) zamiast autograd dla stabilności.
    """
    device = pred_3ch.device

    if detach_u:
        u_x = pred_3ch[:, 0:1, :, :].detach()
        u_y = pred_3ch[:, 1:2, :, :].detach()
    else:
        u_x = pred_3ch[:, 0:1, :, :]
        u_y = pred_3ch[:, 1:2, :, :]

    h = pred_3ch[:, 2:3, :, :]

    # Filtry Sobela do wyznaczania pierwszych pochodnych (gradientów)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3) / (8.0 * dx)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3) / (8.0 * dx)

    dux_dx = F.conv2d(u_x, sobel_x, padding=1)
    dux_dy = F.conv2d(u_x, sobel_y, padding=1)
    duy_dx = F.conv2d(u_y, sobel_x, padding=1)
    duy_dy = F.conv2d(u_y, sobel_y, padding=1)

    dh_dx = F.conv2d(h, sobel_x, padding=1)
    dh_dy = F.conv2d(h, sobel_y, padding=1)

    # Filtr Laplasjanu do obliczenia drugich pochodnych
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3) / (dx**2)
    lap_ux = F.conv2d(u_x, lap_kernel, padding=1)
    lap_uy = F.conv2d(u_y, lap_kernel, padding=1)
    lap_h = F.conv2d(h, lap_kernel, padding=1)

    # Norma gradientu pola pomocniczego u (źródło dla równania Poissona)
    norm_grad_u = torch.sqrt(
        dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2 + 1e-8
    )

    # 1. Strata Równań Różniczkowych (PDE Residual Losses)
    loss_laplace = torch.mean(lap_ux**2) + torch.mean(lap_uy**2)
    loss_poisson = torch.mean((lap_h + norm_grad_u) ** 2)
    pde_loss = loss_laplace + loss_poisson

    # 2. Strata Warunków Brzegowych (BC Losses)
    loss_bc_h = torch.mean((h * grid.float()) ** 2)
    loss_bc_h_outer = (
        torch.mean(h[:, :, 0, :] ** 2)
        + torch.mean(h[:, :, -1, :] ** 2)
        + torch.mean(h[:, :, :, 0] ** 2)
        + torch.mean(h[:, :, :, -1] ** 2)
    )

    boundary_masks = compute_boundary_mask(grid)
    u_targets = torch.cat([ux_true, uy_true], dim=1)
    loss_bc_u = torch.mean(
        ((pred_3ch[:, 0:2] - u_targets) ** 2) * boundary_masks
    )

    bc_loss = loss_bc_h + loss_bc_h_outer + loss_bc_u

    # 3. Strata danych dla h i jego pochodnych
    loss_h_data = F.mse_loss(h, h_true)
    loss_dhdx_data = F.mse_loss(dh_dx, dhdx_true)
    loss_dhdy_data = F.mse_loss(dh_dy, dhdy_true)
    loss_ux_data = F.mse_loss(u_x, ux_true)
    loss_uy_data = F.mse_loss(u_y, uy_true)

    # Add after computing losses, before returning
    # Normalize PDE loss by dx² to make it resolution-independent
    pde_loss = pde_loss * (dx**2)  # Scale back the 1/dx² explosion

    # Normalize gradient data losses similarly
    loss_dhdx_data = loss_dhdx_data * (dx**2)
    loss_dhdy_data = loss_dhdy_data * (dx**2)
    loss_ux_data = loss_ux_data * (dx**2)  
    loss_uy_data = loss_uy_data * (dx**2)
    value_data_loss = (
        loss_h_data
        + loss_dhdx_data
        + loss_dhdy_data
        + loss_ux_data
        + loss_uy_data
    )

    return pde_loss, value_data_loss, bc_loss, dux_dx, dux_dy

def predict_safety_with_gradients(model, x, y):
    """Pozwala na odpytanie sieci h_net o wartość bezpieczeństwa i jej gradienty (model MLP)."""
    pt_tensor = torch.tensor(
        [[x, y]], dtype=torch.float32, requires_grad=True
    )
    h_val = model.h_net(pt_tensor)

    grads = torch.autograd.grad(
        h_val, pt_tensor, grad_outputs=torch.ones_like(h_val)
    )[0]
    dh_dx = grads[0, 0].item()
    dh_dy = grads[0, 1].item()

    return h_val.item(), dh_dx, dh_dy


def plot_poisson_prediction_example(
    model,
    grid,
    grid_size,
    device=None,
    normalization_stats=None,
    h_true=None,
    ux_true=None,
    uy_true=None,
    save_path=None,
    show=False,
):
    """Rysuje przykład predykcji dla jednej mapy testowej lub wejścia z inferencji."""
    if device is None:
        device = next(model.parameters()).device

    def _to_numpy_2d(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        return np.squeeze(array)

    def _maybe_denorm_2d(array_2d, field_name):
        if array_2d is None or normalization_stats is None:
            return array_2d
        if field_name not in normalization_stats:
            return array_2d
        mean = normalization_stats[field_name]["mean"]
        std = normalization_stats[field_name]["std"]
        return array_2d * std + mean

    grid_np = _to_numpy_2d(grid)
    if grid_np.ndim == 3:
        grid_np = np.squeeze(grid_np, axis=0)

    if isinstance(grid, torch.Tensor):
        grid_tensor = grid.detach().clone()
    else:
        grid_tensor = torch.tensor(grid, dtype=torch.float32)

    if grid_tensor.ndim == 2:
        grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)
    elif grid_tensor.ndim == 3:
        grid_tensor = grid_tensor.unsqueeze(0)

    grid_tensor = grid_tensor.to(device)

    model.eval()
    with torch.no_grad():
        preds_3ch = model(grid_tensor)
        if normalization_stats is not None:
            preds_3ch = denormalize_pred_3ch(
                preds_3ch, normalization_stats
            )
        preds_3ch = preds_3ch.cpu().numpy().squeeze(0)

    u_x_pred = preds_3ch[0]
    u_y_pred = preds_3ch[1]
    h_pred = preds_3ch[2]
    magnitude = np.sqrt(u_x_pred**2 + u_y_pred**2)

    mask = grid_np == 0
    u_x_pred = np.where(mask, u_x_pred, np.nan)
    u_y_pred = np.where(mask, u_y_pred, np.nan)
    h_pred = np.where(mask, h_pred, np.nan)
    magnitude = np.where(mask, magnitude, np.nan)

    x = np.linspace(-5, 5, int(grid_size))
    y = np.linspace(-5, 5, int(grid_size))
    X, Y = np.meshgrid(x, y)

    has_reference = (
        h_true is not None and ux_true is not None and uy_true is not None
    )
    fig, axes = plt.subplots(1, 4 if has_reference else 2, figsize=(18, 5.5))

    if has_reference:
        h_true_np = _to_numpy_2d(h_true)
        ux_true_np = _to_numpy_2d(ux_true)
        uy_true_np = _to_numpy_2d(uy_true)

        h_true_np = _maybe_denorm_2d(h_true_np, "h")
        ux_true_np = _maybe_denorm_2d(ux_true_np, "ux")
        uy_true_np = _maybe_denorm_2d(uy_true_np, "uy")

        ux_true_np = np.where(mask, ux_true_np, np.nan)
        uy_true_np = np.where(mask, uy_true_np, np.nan)
        h_true_np = np.where(mask, h_true_np, np.nan)

        if ux_true_np.shape != X.shape and ux_true_np.T.shape == X.shape:
            ux_true_np = ux_true_np.T
            uy_true_np = uy_true_np.T

        ax = axes[0]
        cp1 = ax.contourf(X, Y, magnitude, levels=50, cmap="viridis")
        fig.colorbar(cp1, ax=ax, label="Magnituda ||u||")
        ax.streamplot(
            X,
            Y,
            ux_true_np,
            uy_true_np,
            color="white",
            linewidth=0.8,
            density=1.0,
        )
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        ax = axes[1]
        cp2 = ax.contourf(X, Y, magnitude, levels=50, cmap="viridis")
        fig.colorbar(cp2, ax=ax, label="Magnituda ||u||")
        u_x_plot = u_x_pred
        u_y_plot = u_y_pred
        if u_x_plot.shape != X.shape and u_x_plot.T.shape == X.shape:
            u_x_plot = u_x_plot.T
            u_y_plot = u_y_plot.T
        ax.streamplot(
            X,
            Y,
            u_x_plot,
            u_y_plot,
            color="white",
            linewidth=0.8,
            density=1.0,
        )
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        ax = axes[2]
        cp3 = ax.contourf(X, Y, h_true_np, levels=50, cmap="plasma")
        fig.colorbar(cp3, ax=ax, label="Wartość h_true")
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Referencyjna funkcja bezpieczeństwa\n(Rozwiązanie numeryczne z H5)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        ax = axes[3]
        cp4 = ax.contourf(X, Y, h_pred, levels=50, cmap="plasma")
        fig.colorbar(cp4, ax=ax, label="Wartość h_pred")
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Wyznaczona funkcja bezpieczeństwa $h(x,y)$\n(Predykcja sieci UNetDoublePoisson)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
    else:
        ax = axes[0]
        cp1 = ax.contourf(X, Y, magnitude, levels=50, cmap="viridis")
        fig.colorbar(cp1, ax=ax, label="Magnituda ||u||")
        ax.streamplot(
            X,
            Y,
            u_x_pred,
            u_y_pred,
            color="white",
            linewidth=0.8,
            density=1.0,
        )
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        ax = axes[1]
        cp2 = ax.contourf(X, Y, h_pred, levels=50, cmap="plasma")
        fig.colorbar(cp2, ax=ax, label="Wartość h_pred")
        ax.imshow(
            grid_np,
            origin="lower",
            extent=[-5, 5, -5, 5],
            cmap="gray_r",
            alpha=0.3,
        )
        ax.set_title(
            "Wyznaczona funkcja bezpieczeństwa $h(x,y)$\n(Predykcja sieci UNetDoublePoisson)"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# --- GŁÓWNA PĘTLA TRENINGOWA DLA WSZYSTKICH MAP ---
if __name__ == "__main__":
    H5_FILE_PATH = "data/nik_training_data_512x512_2000.h5"
    WEIGHTS_PATH = "weights/pde_1_0_poisson_unet_bilinear_model_512x512_2000_e40_normalization_bc_fix_test.pth"
    BEST_WEIGHTS_PATH = WEIGHTS_PATH.replace(".pth", "_best.pth")
    NORM_STATS_PATH = WEIGHTS_PATH.replace(".pth", "_norm_stats.pt")
    GRID_SIZE = 512.0
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Używane urządzenie: {device}")

    # 1. Weryfikacja pliku z danymi i ładowanie
    try:
        full_dataset = H5PoissonDataset(H5_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        print(
            "Najpierw wygeneruj zestaw danych przy użyciu skryptu generate_maps_and_psf.py!"
        )
        exit(1)

    # 2. Podział na zbiór treningowy i walidacyjny (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    # Statystyki normalizacji wyliczamy wyłącznie na zbiorze treningowym.
    train_indices = train_dataset.indices
    normalization_stats = compute_normalization_stats(
        full_dataset, train_indices
    )
    full_dataset.set_normalization_stats(normalization_stats)

    os.makedirs(os.path.dirname(NORM_STATS_PATH) or ".", exist_ok=True)
    torch.save(
        {
            "stats": normalization_stats,
            "data_path": H5_FILE_PATH,
            "grid_size": GRID_SIZE,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        NORM_STATS_PATH,
    )
    print(f"Zapisano statystyki normalizacji do {NORM_STATS_PATH}")

    # Batch size ustawiony na 2 ze względu na wysokie zapotrzebowanie RAM/VRAM przy wymiarach GRID_SIZExGRID_SIZE
    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(
        f"Zestaw treningowy: {len(train_dataset)} map | Zestaw walidacyjny: {len(val_dataset)} map"
    )

    # 3. Inicjalizacja sieci splotowej UNetDoublePoisson i optymalizatora
    model = UNetDoublePoisson().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 40
    w_pde = 1.0
    w_bc = 1
    w_mse = 1
    early_stopping_patience = 12
    early_stopping_min_delta = 1e-6
    best_val_total = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    dx = (
        10.0 / GRID_SIZE
    )  # fizyczny krok siatki dla szerokości 10 [-5.0, 5.0]

    print(
        "\nRozpoczynanie treningu splotowej sieci UNetDoublePoisson na wszystkich mapach..."
    )
    prev_epoch = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoka {epoch + 1:02d}/{epochs:02d}",
            leave=True,
        )
        for (
            grid,
            h_true,
            dhdx_true,
            dhdy_true,
            ux_true,
            uy_true,
        ) in pbar:
            grid = grid.to(device)
            h_true = h_true.to(device)
            dhdx_true = dhdx_true.to(device)
            dhdy_true = dhdy_true.to(device)
            ux_true = ux_true.to(device)
            uy_true = uy_true.to(device)

            optimizer.zero_grad()

            # Przejście w przód przez sieć splotową U-Net (zwraca [B, 3, GRID_SIZE, GRID_SIZE])
            pred_3ch = model(grid)
            h_pred = pred_3ch[
                :, 2:3, :, :
            ]  # Kanał 2 to funkcja bezpieczeństwa h

            # Strata danych (porównanie przewidywanego h ze stanem faktycznym z H5)
            loss_data_h = criterion(h_pred, h_true)
            loss_data_ux = criterion(pred_3ch[:, 0:1, :, :], ux_true)
            loss_data_uy = criterion(pred_3ch[:, 1:2, :, :], uy_true)
            loss_data = loss_data_h + loss_data_ux + loss_data_uy

            # Strata fizyczna PDE wyznaczana splotowo na GPU
            pred_phys = denormalize_pred_3ch(pred_3ch, normalization_stats)
            pde_loss, _, bc_loss, _, _ = calc_poisson_pinn_loss(
                pred_phys,
                h_true,
                dhdx_true,
                dhdy_true,
                ux_true,
                uy_true,
                grid,
                dx=dx,
                detach_u=True,
            )
            # Całkowita hybrydowa strata (Dane + Fizyka)
            loss = w_mse*loss_data + w_pde * pde_loss + w_bc * bc_loss
            if prev_epoch != epoch:
                print(f"lmse: {loss_data}, pde: {w_pde * pde_loss}, bc: {w_bc * bc_loss}, total: {loss}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            prev_epoch = epoch

        # --- Walidacja po każdej epoce ---
        model.eval()
        val_mse_h = 0.0
        val_mse_ux = 0.0
        val_mse_uy = 0.0
        val_data_loss = 0.0
        val_pde_loss = 0.0
        val_bc_loss = 0.0
        val_total_loss = 0.0
        val_idx=0
        with torch.no_grad():
            for (
                grid_val,
                h_val_true,
                dhdx_val_true,
                dhdy_val_true,
                ux_val_true,
                uy_val_true,
            ) in val_loader:
                grid_val = grid_val.to(device)
                h_val_true = h_val_true.to(device)
                dhdx_val_true = dhdx_val_true.to(device)
                dhdy_val_true = dhdy_val_true.to(device)
                ux_val_true = ux_val_true.to(device)
                uy_val_true = uy_val_true.to(device)

                pred_val = model(grid_val)
                h_val_pred = pred_val[:, 2:3, :, :]
                val_h = criterion(h_val_pred, h_val_true)
                val_ux = criterion(pred_val[:, 0:1, :, :], ux_val_true)
                val_uy = criterion(pred_val[:, 1:2, :, :], uy_val_true)
                val_data = val_h + val_ux + val_uy

                val_mse_h += val_h.item()
                val_mse_ux += val_ux.item()
                val_mse_uy += val_uy.item()
                val_data_loss += val_data.item()

                pred_val_phys = denormalize_pred_3ch(
                    pred_val, normalization_stats
                )
                pde_val, _, bc_val, _, _ = calc_poisson_pinn_loss(
                    pred_val_phys,
                    h_val_true,
                    dhdx_val_true,
                    dhdy_val_true,
                    ux_val_true,
                    uy_val_true,
                    grid_val,
                    dx=dx,
                    detach_u=True,
                )
                val_pde_item = pde_val.item()
                val_bc_item = bc_val.item()
                val_total_item = (
                    w_mse * val_data.item()
                    + w_pde * val_pde_item
                    + w_bc * val_bc_item
                )

                val_pde_loss += val_pde_item
                val_bc_loss += val_bc_item
                val_total_loss += val_total_item
                
        val_mse_h_avg = val_mse_h / len(val_loader)
        val_mse_ux_avg = val_mse_ux / len(val_loader)
        val_mse_uy_avg = val_mse_uy / len(val_loader)
        val_data_avg = val_data_loss / len(val_loader)
        val_pde_avg = val_pde_loss / len(val_loader)
        val_bc_avg = val_bc_loss / len(val_loader)
        val_total_avg = val_total_loss / len(val_loader)

        print(
            f"-> Epoka {epoch + 1:02d} | Średni Loss Treningowy: {running_loss / len(train_loader):.6f} | "
            f"Walidacja (MSE_h): {val_mse_h_avg:.6f} | Walidacja (MSE_ux): {val_mse_ux_avg:.6f} | "
            f"Walidacja (MSE_uy): {val_mse_uy_avg:.6f} | Walidacja (DATA): {val_data_avg:.6f} | "
            f"Walidacja (PDE): {val_pde_avg:.6f} | Walidacja (BC): {val_bc_avg:.6f} | "
            f"Walidacja (TOTAL): {val_total_avg:.6f}"
        )

        if val_total_avg < best_val_total - early_stopping_min_delta:
            best_val_total = val_total_avg
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            print(
                f"Nowe najlepsze wagi (epoka {best_epoch:02d}) zapisane do {BEST_WEIGHTS_PATH}"
            )
        else:
            epochs_no_improve += 1
            print(
                f"Brak poprawy przez {epochs_no_improve} epok (patience={early_stopping_patience})"
            )
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Wczesne zatrzymanie: brak poprawy Walidacja (TOTAL). Najlepsza epoka: {best_epoch:02d}"
                )
                break

        grid, h_true, dhdx_true, dhdy_true, ux_true, uy_true = val_dataset[0]
        # plot_poisson_prediction_example(
        #     model,
        #     grid,
        #     grid_size=GRID_SIZE,
        #     device=device,
        #     h_true=h_true,
        #     ux_true=ux_true,
        #     uy_true=uy_true,
        #     save_path="fig/poisson_unet_bilinear_model_512x512_1000_e20.png",
        #     show=False,
        # )
        

    # 4. Załadowanie najlepszych wag i zapis końcowy
    if os.path.isfile(BEST_WEIGHTS_PATH):
        model.load_state_dict(
            torch.load(BEST_WEIGHTS_PATH, map_location=device)
        )

    # 5. Zapisanie wag modelu na dysku
    os.makedirs(os.path.dirname(WEIGHTS_PATH) or ".", exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nPomyślnie zapisano wagi modelu do {WEIGHTS_PATH}")

    # --- 6. WIZUALIZACJA WYNIKÓW DLA PIERWSZEJ MAPY TESTOWEJ ---
    print("\nGenerowanie wykresu końcowego dla wybranej mapy...")
    grid, h_true, dhdx_true, dhdy_true, ux_true, uy_true = val_dataset[0]
    plot_poisson_prediction_example(
        model,
        grid,
        grid_size=GRID_SIZE,
        device=device,
        normalization_stats=normalization_stats,
        h_true=h_true,
        ux_true=ux_true,
        uy_true=uy_true,
        save_path="fig/pde_1_0_poisson_unet_bilinear_model_512x512_2000_e40_normalization_bc_fix_test.png",
        show=False,
    )