#!/usr/bin/env python3
"""
Skrypt do generowania map przeszkód i obliczania dokładnych funkcji bezpieczeństwa Poissona (PSF)
przy użyciu sprzężonego solvera różnic skończonych na regularnej siatce w obszarze [-5.0, 5.0]^2.
Domyślna rozdzielczość siatki wynosi 128x128.
"""

from __future__ import annotations

import argparse
import os
import math
import numpy as np
import h5py
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_laplace_and_poisson(objects_list, n=128):
    """
    Rozwiązuje sprzężony układ równań na obszarze [-5, 5]^2:
    1) Delta u = 0 w Omega, u = x - c_i na krawędziach przeszkód
    2) Delta h = -||grad u|| w Omega, h = 0 na krawędziach przeszkód i ścianach zewnętrznych

    - objects_list: lista słowników z kluczami 'c' (środek) i 'r' (promień)
    - n: rozdzielczość siatki (domyślnie 128)
    - zwraca: h (n,n), dhdx (n,n), dhdy (n,n), grid (n,n uint8)
    """
    nx = ny = n
    x = np.linspace(-5.0, 5.0, nx)
    y = np.linspace(-5.0, 5.0, ny)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    # 1. Identyfikacja masek przeszkód i wyznaczenie środka ciężkości dla każdej z nich
    mask = np.zeros((ny, nx), dtype=bool)
    obstacle_id_map = -np.ones((ny, nx), dtype=int)
    centers = {}

    for idx_obj, obj in enumerate(objects_list):
        cx, cy = float(obj["c"][0]), float(obj["c"][1])
        r = float(obj["r"])
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        in_obstacle = dist <= r + 1e-12
        mask |= in_obstacle
        obstacle_id_map[in_obstacle] = idx_obj
        centers[idx_obj] = (cx, cy)

    # 2. Definicja węzłów niewiadomych (poza przeszkodami i poza zewnętrznymi krawędziami)
    unknown = np.ones((ny, nx), dtype=bool)
    unknown[0, :] = False
    unknown[-1, :] = False
    unknown[:, 0] = False
    unknown[:, -1] = False
    unknown[mask] = False

    # Mapa indeksów węzłów niewiadomych na potrzeby solvera rzadkiego
    node_idx = -np.ones((ny, nx), dtype=int)
    idx_flat = 0
    for j in range(ny):
        for i in range(nx):
            if unknown[j, i]:
                node_idx[j, i] = idx_flat
                idx_flat += 1

    N = idx_flat
    if N == 0:
        h = np.zeros((ny, nx), dtype=np.float32)
        return (
            h,
            np.zeros_like(h),
            np.zeros_like(h),
            np.zeros((ny, nx), dtype=np.uint8),
        )

    # --- Budowanie rzadkiej macierzy Laplasjanu A (szablon 5-punktowy) ---
    data = []
    rows = []
    cols = []

    def add_entry(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not unknown[j, i]:
                continue
            row = node_idx[j, i]
            add_entry(row, row, -4.0 / (dx * dx))
            for jj, ii in (
                (j - 1, i),
                (j + 1, i),
                (j, i - 1),
                (j, i + 1),
            ):
                if unknown[jj, ii]:
                    add_entry(row, node_idx[jj, ii], 1.0 / (dx * dx))

    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # --- 1. ROZWIĄZYWANIE RÓWNANIA LAPLACE'A DLA POLA WEKTOROWEGO u ---
    # Rozwiązujemy niezależnie dla składowych u_x oraz u_y
    rhs_ux = np.zeros(N, dtype=np.float64)
    rhs_uy = np.zeros(N, dtype=np.float64)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not unknown[j, i]:
                continue
            row = node_idx[j, i]

            # Warunki brzegowe wnoszą wkład do prawej strony (RHS), jeśli sąsiedzi są krawędziami
            for jj, ii in (
                (j - 1, i),
                (j + 1, i),
                (j, i - 1),
                (j, i + 1),
            ):
                if not unknown[jj, ii]:
                    if mask[jj, ii]:
                        obs_id = obstacle_id_map[jj, ii]
                        cx, cy = centers[obs_id]
                        val_ux = X[jj, ii] - cx
                        val_uy = Y[jj, ii] - cy
                    else:
                        # Zewnętrzne krawędzie obszaru mają zerowy potencjał
                        val_ux = 0.0
                        val_uy = 0.0

                    rhs_ux[row] -= val_ux / (dx * dx)
                    rhs_uy[row] -= val_uy / (dx * dx)

    u_x_sol = spsolve(A, rhs_ux)
    u_y_sol = spsolve(A, rhs_uy)

    # Odtwarzanie pełnych pól składowych u
    u_x = np.zeros((ny, nx), dtype=np.float32)
    u_y = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            if unknown[j, i]:
                u_x[j, i] = u_x_sol[node_idx[j, i]]
                u_y[j, i] = u_y_sol[node_idx[j, i]]
            elif mask[j, i]:
                obs_id = obstacle_id_map[j, i]
                cx, cy = centers[obs_id]
                u_x[j, i] = X[j, i] - cx
                u_y[j, i] = Y[j, i] - cy

    # Numeryczne obliczanie gradientów pola u
    duy_dy, duy_dx = np.gradient(u_y, y, x)
    dux_dy, dux_dx = np.gradient(u_x, y, x)

    # Wyznaczenie modułu gradientu (człon źródłowy równania Poissona): f = ||grad u||
    norm_grad_u = np.sqrt(
        dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2 + 1e-8
    )

    # --- 2. ROZWIĄZYWANIE RÓWNANIA POISSONA DLA FUNKCJI BEZPIECZEŃSTWA h ---
    # Równanie: Delta h = -||grad u||  =>  A * h = -norm_grad_u
    rhs_h = np.zeros(N, dtype=np.float64)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not unknown[j, i]:
                continue
            row = node_idx[j, i]
            rhs_h[row] = -float(norm_grad_u[j, i])

    h_sol = spsolve(A, rhs_h)

    # Odtwarzanie pełnego pola h
    h = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            if unknown[j, i]:
                h[j, i] = h_sol[node_idx[j, i]]

    # Obliczanie pochodnych przestrzennych funkcji h
    dhdy, dhdx = np.gradient(h, y, x)

    # Czyszczenie i filtracja wartości nieliczbowych (NaN)
    h = np.nan_to_num(h, nan=0.0)
    dhdx = np.nan_to_num(dhdx, nan=0.0)
    dhdy = np.nan_to_num(dhdy, nan=0.0)
    grid = mask.astype(np.uint8)

    return u_x, u_y, h, dhdx, dhdy, grid


def make_objects_list_for_index(map_index, max_objects_number=12):
    """Generuje losowe rozłożenie przeszkód wewnątrz obszaru [-5, 5]^2."""
    rng = np.random.RandomState(map_index)
    objects_number = int(math.ceil(max_objects_number * rng.rand()))
    objects_list = []

    if map_index % 20 == 0:
        return objects_list

    for _ in range(objects_number):
        attempts = 0
        max_attempts = 200
        valid = False
        while not valid and attempts < max_attempts:
            attempts += 1
            # Pozycja losowa w zakresie [-4.3, 4.3] dla zachowania marginesu bezpieczeństwa
            potential_c = rng.uniform(-4.3, 4.3, 2)
            potential_r = rng.uniform(
                0.3, 0.5
            )  # Zrównoważony promień przeszkód

            overlap = False
            for obj in objects_list:
                dist_centers = np.linalg.norm(
                    potential_c - np.array(obj["c"])
                )
                if dist_centers < (potential_r + obj["r"] + 0.4):
                    overlap = True
                    break

            if not overlap:
                valid = True
                objects_list.append(
                    {"c": potential_c.copy(), "r": float(potential_r)}
                )

    return objects_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", nargs="?", default="training_data_128x128.h5"
    )
    parser.add_argument(
        "generated_maps_number", nargs="?", type=int, default=2000
    )
    parser.add_argument(
        "resolution", nargs="?", type=int, default=128
    )
    args = parser.parse_args()

    file_name = args.file_name
    generated_maps_number = args.generated_maps_number
    res = args.resolution

    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    for map_index in range(1, generated_maps_number + 1):
        objects_list = make_objects_list_for_index(map_index)

        u_x, u_y, h, dhdx, dhdy, grid = solve_laplace_and_poisson(
            objects_list, n=res
        )

        index_string = f"{map_index:06d}"
        with h5py.File(file_name, "a") as f:
            # Zapisujemy macierze o kształcie [1, H, W] zgodnym z architekturami splotowymi PyTorch
            grid_data = np.expand_dims(grid, axis=0).astype(np.uint8)
            h_data = np.expand_dims(h, axis=0).astype(np.float32)
            u_x_data = np.expand_dims(u_x, axis=0).astype(np.float32)
            u_y_data = np.expand_dims(u_y, axis=0).astype(np.float32)
            dhdx_data = np.expand_dims(dhdx, axis=0).astype(
                np.float32
            )
            dhdy_data = np.expand_dims(dhdy, axis=0).astype(
                np.float32
            )

            grp_grid = f.require_group("grid")
            if index_string in grp_grid:
                del grp_grid[index_string]
            grp_grid.create_dataset(
                index_string, data=grid_data, dtype="u1"
            )

            grp_u_x = f.require_group("u_x")
            if index_string in grp_u_x:
                del grp_u_x[index_string]
            grp_u_x.create_dataset(
                index_string, data=u_x_data, dtype="f4"
            )

            grp_u_y = f.require_group("u_y")
            if index_string in grp_u_y:
                del grp_u_y[index_string]
            grp_u_y.create_dataset(
                index_string, data=u_y_data, dtype="f4"
            )

            grp_h = f.require_group("h")
            if index_string in grp_h:
                del grp_h[index_string]
            grp_h.create_dataset(
                index_string, data=h_data, dtype="f4"
            )

            grp_dhdx = f.require_group("dhdx")
            if index_string in grp_dhdx:
                del grp_dhdx[index_string]
            grp_dhdx.create_dataset(
                index_string, data=dhdx_data, dtype="f4"
            )

            grp_dhdy = f.require_group("dhdy")
            if index_string in grp_dhdy:
                del grp_dhdy[index_string]
            grp_dhdy.create_dataset(
                index_string, data=dhdy_data, dtype="f4"
            )

        if map_index % 50 == 0:
            print(
                f"Wygenerowano sprzężoną mapę {map_index}/{generated_maps_number} ({res}x{res}) -> {file_name}:{index_string}"
            )


if __name__ == "__main__":
    main()
