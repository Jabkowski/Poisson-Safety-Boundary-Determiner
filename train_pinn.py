import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.ndimage import binary_erosion, label, center_of_mass
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Ustawienie ziarna losowości dla powtarzalności wyników
torch.manual_seed(42)
np.random.seed(42)


# class DoublePoissonPINN(nn.Module):
#     """
#     PINN (MLP) zawierający DWIE osobne sieci neuronowe:
#     1. u_net: wejście (x, y) -> wyjście (u_x, u_y) [Pole Laplace'a]
#     2. h_net: wejście (x, y) -> wyjście (h)       [Funkcja bezpieczeństwa Poissona]
#     Dzięki temu eliminujemy konflikt gradientów między dwoma różnymi równaniami fizycznymi.
#     """

#     def __init__(self):
#         super(DoublePoissonPINN, self).__init__()

#         # Sieć dla pola pomocniczego u (rozwiązuje układ Laplace'a)
#         self.u_net = nn.Sequential(
#             nn.Linear(2, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 2),  # Wyjścia: u_x, u_y
#         )

#         # Sieć dla funkcji bezpieczeństwa h (rozwiązuje równanie Poissona)
#         self.h_net = nn.Sequential(
#             nn.Linear(2, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1),  # Wyjście: h
#         )

#     def forward(self, x):
#         """Zwraca spójny tensor 3-kanałowy [u_x, u_y, h] dla wstecznej kompatybilności"""
#         u = self.u_net(x)
#         h = self.h_net(x)
#         return torch.cat([u, h], dim=1)


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
        self.u_net = UNetSubNetwork(in_channels=1, out_channels=2)
        self.h_net = UNetSubNetwork(in_channels=2, out_channels=1)

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

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self.h_values[idx],
            self.dhdx_values[idx],
            self.dhdy_values[idx],
            self.ux_values[idx],
            self.uy_values[idx],
        )


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

        # # Budowanie celu wektorowego na krawędziach
        # rows, cols = np.where(boundary_np)
        # for r, c in zip(rows, cols):
        #     wx = -5.0 + (c / (W - 1)) * 10.0
        #     wy = -5.0 + (r / (H - 1)) * 10.0
        #     obs_id = labeled[r, c]
        #     if obs_id in centers:
        #         cx, cy = centers[obs_id]
        #         U_targets[b, 0, r, c] = 1.0 * (wx - cx)  # u_x_target
        #         U_targets[b, 1, r, c] = 1.0 * (wy - cy)  # u_y_target

    return U_targets, boundary_masks


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

    u_targets, boundary_masks = compute_batch_boundary_u_targets(
        grid, dx
    )
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

    value_data_loss = (
        loss_h_data
        + loss_dhdx_data
        + loss_dhdy_data
        + loss_ux_data
        + loss_uy_data
    )

    return pde_loss, value_data_loss, bc_loss, dux_dx, dux_dy


# def calc_poisson_pinn_loss_old(
#     pred_3ch, grid, dx=10.0 / 128.0, detach_u=True
# ):
#     """
#     Oblicza fizyczny błąd PDE (Physics-Informed Loss) przy użyciu splotów 2D dla paczki (Batch).
#     detach_u=True blokuje przepływ wsteczny gradientów h do podsieci pola pomocniczego u.
#     """
#     device = pred_3ch.device

#     if detach_u:
#         u_x = pred_3ch[:, 0:1, :, :].detach()
#         u_y = pred_3ch[:, 1:2, :, :].detach()
#     else:
#         u_x = pred_3ch[:, 0:1, :, :]
#         u_y = pred_3ch[:, 1:2, :, :]

#     h = pred_3ch[:, 2:3, :, :]

#     # Filtry Sobela do wyznaczania pierwszych pochodnych (gradientów)
#     sobel_x = torch.tensor(
#         [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#         dtype=torch.float32,
#         device=device,
#     ).view(1, 1, 3, 3) / (8.0 * dx)
#     sobel_y = torch.tensor(
#         [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#         dtype=torch.float32,
#         device=device,
#     ).view(1, 1, 3, 3) / (8.0 * dx)

#     dux_dx = F.conv2d(u_x, sobel_x, padding=1)
#     dux_dy = F.conv2d(u_x, sobel_y, padding=1)
#     duy_dx = F.conv2d(u_y, sobel_x, padding=1)
#     duy_dy = F.conv2d(u_y, sobel_y, padding=1)

#     # Filtr Laplasjanu do obliczenia drugich pochodnych
#     lap_kernel = torch.tensor(
#         [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
#         dtype=torch.float32,
#         device=device,
#     ).view(1, 1, 3, 3) / (dx**2)
#     lap_ux = F.conv2d(u_x, lap_kernel, padding=1)
#     lap_uy = F.conv2d(u_y, lap_kernel, padding=1)
#     lap_h = F.conv2d(h, lap_kernel, padding=1)

#     # Norma gradientu pola pomocniczego u (źródło dla równania Poissona)
#     norm_grad_u = torch.sqrt(
#         dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2 + 1e-8
#     )

#     # 1. Strata Równań Różniczkowych (PDE Residual Losses)
#     loss_laplace = torch.mean(lap_ux**2) + torch.mean(lap_uy**2)
#     loss_poisson = torch.mean((lap_h + norm_grad_u) ** 2)
#     pde_loss = loss_laplace + loss_poisson

#     # 2. Strata Warunków Brzegowych (BC Losses)
#     loss_bc_h = torch.mean((h * grid.float()) ** 2)
#     loss_bc_h_outer = (
#         torch.mean(h[:, :, 0, :] ** 2)
#         + torch.mean(h[:, :, -1, :] ** 2)
#         + torch.mean(h[:, :, :, 0] ** 2)
#         + torch.mean(h[:, :, :, -1] ** 2)
#     )

#     u_targets, boundary_masks = compute_batch_boundary_u_targets(
#         grid, dx
#     )
#     loss_bc_u = torch.mean(
#         ((pred_3ch[:, 0:2] - u_targets) ** 2) * boundary_masks
#     )

#     bc_loss = loss_bc_h + loss_bc_h_outer + loss_bc_u

#     return pde_loss, bc_loss, dux_dx, dux_dy


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


# --- GŁÓWNA PĘTLA TRENINGOWA DLA WSZYSTKICH MAP ---
if __name__ == "__main__":
    H5_FILE_PATH = "data/nik_training_data_512x512_2000.h5"
    WEIGHTS_PATH = "weights/double_poisson_unet_model_e40.pth"
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

    # Batch size ustawiony na 2 ze względu na wysokie zapotrzebowanie RAM/VRAM przy wymiarach GRID_SIZExGRID_SIZE
    batch_size = 8
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
    w_pde = 0.01
    w_bc = 0.1
    dx = (
        10.0 / GRID_SIZE
    )  # fizyczny krok siatki dla szerokości 10 [-5.0, 5.0]

    print(
        "\nRozpoczynanie treningu splotowej sieci UNetDoublePoisson na wszystkich mapach..."
    )
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
            pde_loss, _, bc_loss, _, _ = calc_poisson_pinn_loss(
                pred_3ch,
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
            loss = loss_data + w_pde * pde_loss + w_bc * bc_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # --- Walidacja po każdej epoce ---
        model.eval()
        val_loss = 0.0
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
                val_loss += criterion(h_val_pred, h_val_true).item()

        print(
            f"-> Epoka {epoch + 1:02d} | Średni Loss Treningowy: {running_loss / len(train_loader):.6f} | Walidacja (MSE): {val_loss / len(val_loader):.6f}"
        )
        

    # 4. Zapisanie wag modelu na dysku
    os.makedirs(os.path.dirname(WEIGHTS_PATH) or ".", exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nPomyślnie zapisano wagi modelu do {WEIGHTS_PATH}")

    # --- 5. WIZUALIZACJA WYNIKÓW DLA PIERWSZEJ MAPY TESTOWEJ ---
    print("\nGenerowanie wykresu końcowego dla wybranej mapy...")
    grid, h_true, dhdx_true, dhdy_true, ux_true, uy_true = (
        val_dataset[0]
    )
    grid_tensor = grid.unsqueeze(0).to(
        device
    )  # Dodanie wymiaru batcha [1, 1, GRID_SIZE, GRID_SIZE]

    model.eval()
    with torch.no_grad():
        preds_3ch = (
            model(grid_tensor).cpu().numpy().squeeze(0)
        )  # [3, GRID_SIZE, GRID_SIZE]

    u_x_pred = preds_3ch[0]
    u_y_pred = preds_3ch[1]
    h_pred = preds_3ch[2]

    grid_np = grid.squeeze().numpy()
    h_true_np = h_true.squeeze().numpy()
    magnitude = np.sqrt(u_x_pred**2 + u_y_pred**2)

    # Zamaskowanie wnętrza przeszkód do rysowania
    mask = grid_np == 0
    u_x_pred[~mask] = np.nan
    u_y_pred[~mask] = np.nan
    h_pred[~mask] = np.nan
    magnitude[~mask] = np.nan

    x = np.linspace(-5, 5, int(GRID_SIZE))
    y = np.linspace(-5, 5, int(GRID_SIZE))
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(18, 5.5))

    # Panel 0: Harmonijny potencjał u (referencyjne)
    plt.subplot(1, 4, 1)
    cp1 = plt.contourf(X, Y, magnitude, levels=50, cmap="viridis")
    plt.colorbar(cp1, label="Magnituda ||u||")
    # Przygotuj wektory referencyjne z dataset (zamień na numpy i dopasuj kształt)
    ux_true_np = (
        ux_true.squeeze().cpu().numpy()
        if isinstance(ux_true, torch.Tensor)
        else np.array(ux_true)
    )
    uy_true_np = (
        uy_true.squeeze().cpu().numpy()
        if isinstance(uy_true, torch.Tensor)
        else np.array(uy_true)
    )
    # Zamaskowanie obszarów poza wolną przestrzenią
    ux_true_np[~mask] = np.nan
    uy_true_np[~mask] = np.nan
    # Transponuj je gdy trzeba, aby dopasować X,Y
    if ux_true_np.shape != X.shape:
        if ux_true_np.T.shape == X.shape:
            ux_true_np = ux_true_np.T
            uy_true_np = uy_true_np.T
    plt.streamplot(
        X,
        Y,
        ux_true_np,
        uy_true_np,
        color="white",
        linewidth=0.8,
        density=1.0,
    )
    plt.imshow(
        grid_np,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.3,
    )
    plt.title(
        "Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    # Panel 1: Harmonijny potencjał u
    plt.subplot(1, 4, 2)
    cp1 = plt.contourf(X, Y, magnitude, levels=50, cmap="viridis")
    plt.colorbar(cp1, label="Magnituda ||u||")
    # Streamplot do wizualizacji kierunków
    # Upewnij się, że macierze wektorów mają ten sam kształt co X, Y
    u_x_plot = u_x_pred
    u_y_plot = u_y_pred
    if u_x_plot.shape != X.shape:
        if u_x_plot.T.shape == X.shape:
            u_x_plot = u_x_plot.T
            u_y_plot = u_y_plot.T
    # Co 4 punkty do poprawnego rysowania wektorów
    plt.streamplot(
        X,
        Y,
        u_x_plot,
        u_y_plot,
        color="white",
        linewidth=0.8,
        density=1.0,
    )
    plt.imshow(
        grid_np,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.3,
    )
    plt.title(
        "Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    # Panel 2: Referencyjne H z pliku H5
    plt.subplot(1, 4, 3)
    cp2 = plt.contourf(X, Y, h_true_np, levels=50, cmap="plasma")
    plt.colorbar(cp2, label="Wartość h_true")
    plt.imshow(
        grid_np,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.3,
    )
    plt.title(
        "Referencyjna funkcja bezpieczeństwa\n(Rozwiązanie numeryczne z H5)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    # Panel 3: Predykcja sieci neuronowej H_pred
    plt.subplot(1, 4, 4)
    cp3 = plt.contourf(X, Y, h_pred, levels=50, cmap="plasma")
    plt.colorbar(cp3, label="Wartość h_pred")
    plt.imshow(
        grid_np,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.3,
    )
    plt.title(
        "Wyznaczona funkcja bezpieczeństwa $h(x,y)$\n(Predykcja sieci UNetDoublePoisson)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig("fig/unet_double_poisson_nikodemus.png", dpi=300)
