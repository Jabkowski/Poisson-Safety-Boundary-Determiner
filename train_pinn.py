import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.ndimage import binary_erosion, label, center_of_mass

# Ustawienie ziarna losowości dla powtarzalności wyników
torch.manual_seed(42)
np.random.seed(42)


class DoublePoissonPINN(nn.Module):
    """
    PINN (MLP) zawierający DWIE osobne sieci neuronowe:
    1. u_net: wejście (x, y) -> wyjście (u_x, u_y) [Pole Laplace'a]
    2. h_net: wejście (x, y) -> wyjście (h)       [Funkcja bezpieczeństwa Poissona]
    Dzięki temu eliminujemy konflikt gradientów między dwoma różnymi równaniami fizycznymi.
    """

    def __init__(self):
        super(DoublePoissonPINN, self).__init__()

        # Sieć dla pola pomocniczego u (rozwiązuje układ Laplace'a)
        self.u_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2),  # Wyjścia: u_x, u_y
        )

        # Sieć dla funkcji bezpieczeństwa h (rozwiązuje równanie Poissona)
        self.h_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),  # Wyjście: h
        )

    def forward(self, x):
        """Zwraca spójny tensor 3-kanałowy [u_x, u_y, h] dla wstecznej kompatybilności"""
        u = self.u_net(x)
        h = self.h_net(x)
        return torch.cat([u, h], dim=1)


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
        self.h_net = UNetSubNetwork(in_channels=1, out_channels=1)

    def forward(self, x):
        u = self.u_net(x)
        h = self.h_net(x)
        return torch.cat([u, h], dim=1)


def compute_batch_boundary_u_targets(grid_batch, dx=10.0 / 512.0):
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

        # Budowanie celu wektorowego na krawędziach
        rows, cols = np.where(boundary_np)
        for r, c in zip(rows, cols):
            wx = -5.0 + (c / (W - 1)) * 10.0
            wy = -5.0 + (r / (H - 1)) * 10.0
            obs_id = labeled[r, c]
            if obs_id in centers:
                cx, cy = centers[obs_id]
                U_targets[b, 0, r, c] = 1.0 * (wx - cx)  # u_x_target
                U_targets[b, 1, r, c] = 1.0 * (wy - cy)  # u_y_target

    return U_targets, boundary_masks


def calc_poisson_pinn_loss(
    pred_3ch, grid, dx=10.0 / 512.0, detach_u=True
):
    """
    Oblicza fizyczny błąd PDE (Physics-Informed Loss) przy użyciu splotów 2D.
    detach_u=True pozwala na zablokowanie przepływu gradientów z równania Poissona (h)
    do sieci generującej u, co zapobiega zniekształceniom pola u.
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

    return pde_loss, bc_loss, dux_dx, dux_dy


def load_grid_from_h5(file_path, map_index=1):
    """Wczytuje konkretną siatkę zajętości z pliku wygenerowanego przez generator H5."""
    index_string = f"{map_index:06d}"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku bazy danych: {file_path}. Wygeneruj go najpierw."
        )

    with h5py.File(file_path, "r") as f:
        grid = np.array(
            f["grid"][index_string], dtype=np.uint8
        ).squeeze()
    return grid


def generate_pinn_data_from_grid(grid):
    """Ekstrahuje punkty treningowe bezpośrednio z wczytanej siatki zajętości."""
    resolution = grid.shape[0]

    def to_world_coords(row, col):
        x = -5.0 + (col / (resolution - 1)) * 10.0
        y = -5.0 + (row / (resolution - 1)) * 10.0
        return x, y

    # Wykrywanie krawędzi przeszkód za pomocą morfologii
    eroded_grid = binary_erosion(grid)
    boundary_grid = grid ^ eroded_grid

    # Segmentacja przeszkód i wyznaczenie ich środków ciężkości
    labeled_array, num_features = label(grid)
    centers = {}
    for i in range(1, num_features + 1):
        row_c, col_c = center_of_mass(labeled_array == i)
        centers[i] = to_world_coords(row_c, col_c)

    # Przygotowanie punktów brzegowych i warunków Dirichleta dla pola u
    boundary_rows, boundary_cols = np.where(boundary_grid == 1)
    X_bc_list = []
    U_bc_list = []
    scale_c = 1.0

    for r, c in zip(boundary_rows, boundary_cols):
        wx, wy = to_world_coords(r, c)
        obstacle_id = labeled_array[r, c]
        cx, cy = centers[obstacle_id]

        u_val_x = scale_c * (wx - cx)
        u_val_y = scale_c * (wy - cy)

        X_bc_list.append([wx, wy])
        U_bc_list.append([u_val_x, u_val_y])

    X_bc = np.array(X_bc_list)
    U_bc = np.array(U_bc_list)

    # Losowanie punktów kolokacji we wnętrzu wolnej przestrzeni (grid == 0)
    raw_points = np.random.uniform(-5.0, 5.0, (15000, 2))
    valid_pde_points = []
    for pt in raw_points:
        col_idx = int((pt[0] + 5.0) / 10.0 * (resolution - 1))
        row_idx = int((pt[1] + 5.0) / 10.0 * (resolution - 1))
        col_idx = np.clip(col_idx, 0, resolution - 1)
        row_idx = np.clip(row_idx, 0, resolution - 1)

        if grid[row_idx, col_idx] == 0:
            valid_pde_points.append(pt)

    X_pde = np.array(valid_pde_points)

    # Punkty zewnętrznych ścian obszaru roboczego (wymuszamy h = 0 na krawędziach świata)
    s = np.linspace(-5.0, 5.0, 200)
    top = np.stack([s, np.full_like(s, 5.0)], axis=1)
    bottom = np.stack([s, np.full_like(s, -5.0)], axis=1)
    left = np.stack([np.full_like(s, -5.0), s], axis=1)
    right = np.stack([np.full_like(s, 5.0), s], axis=1)
    X_outer = np.vstack([top, bottom, left, right])

    X_pde_tensor = torch.tensor(
        X_pde, dtype=torch.float32, requires_grad=True
    )
    X_bc_tensor = torch.tensor(X_bc, dtype=torch.float32)
    U_bc_tensor = torch.tensor(U_bc, dtype=torch.float32)
    X_outer_tensor = torch.tensor(X_outer, dtype=torch.float32)

    return (
        X_pde_tensor,
        X_bc_tensor,
        U_bc_tensor,
        X_outer_tensor,
        grid,
    )


def compute_joint_pde_residuals(model, x_pde, detach_u=True):
    """
    Oblicza błędy residualne przy użyciu Autogradu w jednym kroku.
    Używa dwóch osobnych podsieci wewnątrz modelu.
    """
    # Predykcja pola u z sieci u_net
    u_out = model.u_net(x_pde)
    u_x = u_out[:, 0:1]
    u_y = u_out[:, 1:2]

    # Obliczamy gradienty u_x i u_y na potrzeby równania Laplace'a
    grad_ux = torch.autograd.grad(
        u_x,
        x_pde,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]
    dux_dx = grad_ux[:, 0:1]
    dux_dy = grad_ux[:, 1:2]

    grad_uy = torch.autograd.grad(
        u_y,
        x_pde,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
    )[0]
    duy_dx = grad_uy[:, 0:1]
    duy_dy = grad_uy[:, 1:2]

    # Obliczamy drugie pochodne dla Laplace'a
    dux_dxx = torch.autograd.grad(
        dux_dx,
        x_pde,
        grad_outputs=torch.ones_like(dux_dx),
        create_graph=True,
    )[0][:, 0:1]
    dux_dyy = torch.autograd.grad(
        dux_dy,
        x_pde,
        grad_outputs=torch.ones_like(dux_dy),
        create_graph=True,
    )[0][:, 1:2]
    laplacian_ux = dux_dxx + dux_dyy

    duy_dxx = torch.autograd.grad(
        duy_dx,
        x_pde,
        grad_outputs=torch.ones_like(duy_dx),
        create_graph=True,
    )[0][:, 0:1]
    duy_dyy = torch.autograd.grad(
        duy_dy,
        x_pde,
        grad_outputs=torch.ones_like(duy_dy),
        create_graph=True,
    )[0][:, 1:2]
    laplacian_uy = duy_dxx + duy_dyy

    # Zapobieganie przepływowi gradientów z h do sieci u_net (jeśli wybrane)
    if detach_u:
        dux_dx_h = dux_dx.detach()
        dux_dy_h = dux_dy.detach()
        duy_dx_h = duy_dx.detach()
        duy_dy_h = duy_dy.detach()
    else:
        dux_dx_h, dux_dy_h, duy_dx_h, duy_dy_h = (
            dux_dx,
            dux_dy,
            duy_dx,
            duy_dy,
        )

    # Predykcja pola h z sieci h_net
    h_val = model.h_net(x_pde)

    grad_h = torch.autograd.grad(
        h_val,
        x_pde,
        grad_outputs=torch.ones_like(h_val),
        create_graph=True,
    )[0]
    dh_dx = grad_h[:, 0:1]
    dh_dy = grad_h[:, 1:2]

    dh_dxx = torch.autograd.grad(
        dh_dx,
        x_pde,
        grad_outputs=torch.ones_like(dh_dx),
        create_graph=True,
    )[0][:, 0:1]
    dh_dyy = torch.autograd.grad(
        dh_dy,
        x_pde,
        grad_outputs=torch.ones_like(dh_dy),
        create_graph=True,
    )[0][:, 1:2]
    lap_h = dh_dxx + dh_dyy

    norm_grad_u = torch.sqrt(
        dux_dx_h**2 + dux_dy_h**2 + duy_dx_h**2 + duy_dy_h**2 + 1e-8
    )
    pde_res_h = lap_h + norm_grad_u

    return laplacian_ux, laplacian_uy, pde_res_h


def predict_safety_with_gradients(model, x, y):
    """Pozwala na odpytanie sieci h_net o wartość bezpieczeństwa i jej gradienty."""
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


# --- GŁÓWNA PĘTLA TRENINGOWA DLA POJEDYNCZEJ MAPY ---
if __name__ == "__main__":
    H5_FILE_PATH = "training_data_512x512.h5"
    MAP_INDEX_TO_TRAIN = 1

    print(
        f"Wczytywanie mapy zajętości o indeksie {MAP_INDEX_TO_TRAIN} z pliku {H5_FILE_PATH}..."
    )
    try:
        grid = load_grid_from_h5(H5_FILE_PATH, MAP_INDEX_TO_TRAIN)
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        print(
            "Generowanie tymczasowej mapy zastępczej do celów demonstracyjnych..."
        )
        resolution = 512
        grid = np.zeros((resolution, resolution), dtype=np.uint8)
        Y, X = np.ogrid[:resolution, :resolution]
        dist_from_center = np.sqrt(
            (X - resolution / 2) ** 2 + (Y - resolution / 2) ** 2
        )
        grid[dist_from_center <= resolution * 0.15] = 1

    print(
        "Przetwarzanie geometrii mapy i generowanie punktów kolokacji..."
    )
    X_pde, X_bc, U_bc, X_outer, grid = generate_pinn_data_from_grid(
        grid
    )

    # Inicjalizacja naszej nowej podwójnej sieci neuronowej
    model = DoublePoissonPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2000
    w_pde_u = 1.0
    w_pde_h = 1.0
    w_bc_u = 20.0
    w_bc_h = 20.0

    print(
        f"Rozpoczęcie ciągłego treningu PINN... Punkty krawędziowe: {X_bc.shape[0]}, Punkty kolokacji: {X_pde.shape[0]}"
    )
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Przejście w przód przez obie podsieci
        U_bc_pred = model.u_net(X_bc)
        h_bc_pred = model.h_net(X_bc)

        loss_bc_u = torch.mean((U_bc_pred - U_bc) ** 2)
        loss_bc_h_obstacles = torch.mean(h_bc_pred**2)

        h_outer_pred = model.h_net(X_outer)
        loss_bc_h_outer = torch.mean(h_outer_pred**2)

        loss_bc = w_bc_u * loss_bc_u + w_bc_h * (
            loss_bc_h_obstacles + loss_bc_h_outer
        )

        # Obliczanie błędów z opcjonalnym odcięciem gradientów pola u (detach_u=True)
        res_ux, res_uy, res_h = compute_joint_pde_residuals(
            model, X_pde, detach_u=True
        )
        loss_pde = w_pde_u * (
            torch.mean(res_ux**2) + torch.mean(res_uy**2)
        ) + w_pde_h * torch.mean(res_h**2)

        total_loss = loss_bc + loss_pde

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | Total Loss: {total_loss.item():.6e} | BC Loss: {loss_bc.item():.6e} | PDE Loss: {loss_pde.item():.6e}"
            )

    # --- WIZUALIZACJA WYNIKÓW ---
    print("\nGenerowanie wykresów wynikowych...")
    resolution = grid.shape[0]

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    mask = np.zeros(grid_points.shape[0], dtype=bool)
    for i, pt in enumerate(grid_points):
        col = int((pt[0] + 5.0) / 10.0 * (resolution - 1))
        row = int((pt[1] + 5.0) / 10.0 * (resolution - 1))
        row = np.clip(row, 0, resolution - 1)
        col = np.clip(col, 0, resolution - 1)
        if grid[row, col] == 0:
            mask[i] = True

    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid_tensor).numpy()

    preds[~mask] = np.nan
    U_x_grid = preds[:, 0].reshape(X.shape)
    U_y_grid = preds[:, 1].reshape(X.shape)
    H_grid = preds[:, 2].reshape(X.shape)
    Magnitude = np.sqrt(U_x_grid**2 + U_y_grid**2)

    H_grad_y, H_grad_x = np.gradient(
        np.where(np.isnan(H_grid), 0, H_grid), 10.0 / 200.0
    )
    H_grad_x[~mask.reshape(X.shape)] = np.nan
    H_grad_y[~mask.reshape(X.shape)] = np.nan

    plt.figure(figsize=(18, 5.5))

    plt.subplot(1, 3, 1)
    cp1 = plt.contourf(X, Y, Magnitude, levels=50, cmap="viridis")
    plt.colorbar(cp1, label="Magnituda ||u||")
    plt.streamplot(
        X,
        Y,
        U_x_grid,
        U_y_grid,
        color="white",
        linewidth=0.8,
        density=1.0,
    )
    plt.imshow(
        grid,
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

    plt.subplot(1, 3, 2)
    cp2 = plt.contourf(X, Y, H_grid, levels=50, cmap="plasma")
    plt.colorbar(cp2, label="Wartość h(x,y)")
    plt.imshow(
        grid,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.3,
    )
    plt.title(
        "Wyznaczona funkcja bezpieczeństwa $h(x,y)$\n(Poisson Safety Function)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, H_grid, levels=30, cmap="plasma", alpha=0.6)
    skip = 10
    plt.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        H_grad_x[::skip, ::skip],
        H_grad_y[::skip, ::skip],
        color="white",
        scale=50,
        width=0.004,
    )
    plt.imshow(
        grid,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="gray_r",
        alpha=0.4,
    )
    plt.title(
        "Wektory gradientu $\\nabla h = [\\partial h/\\partial x, \\partial h/\\partial y]^T$\n(Kierunki najszybszego wzrostu bezpieczeństwa)"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()
