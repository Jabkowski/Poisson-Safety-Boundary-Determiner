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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import h5py
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import splu
import numpy as np


def solve_laplace_and_poisson(
    objects_list,
    n=128,
):
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

    def compute_object_mask(obj):
        cx, cy = float(obj["c"][0]), float(obj["c"][1])
        if obj.get("type", "circle") in ("rect", "rectangle"):
            hx, hy = float(obj["size"][0]), float(obj["size"][1])
            return (np.abs(X - cx) <= hx + 1e-12) & (
                np.abs(Y - cy) <= hy + 1e-12
            )

        r = float(obj["r"])
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        return dist <= r + 1e-12

    for idx_obj, obj in enumerate(objects_list):
        cx, cy = float(obj["c"][0]), float(obj["c"][1])
        in_obstacle = compute_object_mask(obj)
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

    # Mapa indeksów węzłów niewiadomych na potrzeby solvera rzadkiego (wektoryzowane)
    unknown_flat = unknown.reshape(-1)
    known_flat = ~unknown_flat
    N = int(unknown_flat.sum())

    if N == 0:
        h = np.zeros((ny, nx), dtype=np.float32)
        return (
            h,
            np.zeros_like(h),
            np.zeros_like(h),
            np.zeros((ny, nx), dtype=np.uint8),
        )

    # --- Budowanie rzadkiej macierzy Laplasjanu A (szablon 5-punktowy), wektoryzowane ---
    # 1D operator drugiej pochodnej (rozmiar n); łączymy przez iloczyn Kroneckera,
    # co odtwarza dokładnie ten sam szablon 5-punktowy co pętle, bez narzutu Pythona.
    inv_dx2 = 1.0 / (dx * dx)
    main_diag = -2.0 * inv_dx2 * np.ones(n)
    off_diag = inv_dx2 * np.ones(n - 1)
    D2 = sparse.diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr",
    )
    I_n = sparse.identity(n, format="csr")
    # Laplasjan pełnej siatki (indeks płaski: idx = j * nx + i)
    A_full = sparse.kron(I_n, D2, format="csr") + sparse.kron(
        D2, I_n, format="csr"
    )

    A = A_full[unknown_flat][:, unknown_flat].tocsc()
    # Blok macierzy łączący węzły niewiadome ze znanymi (dla wkładu warunków brzegowych do RHS)
    A_bc = A_full[unknown_flat][:, known_flat]

    # --- 1. ROZWIĄZYWANIE RÓWNANIA LAPLACE'A DLA POLA WEKTOROWEGO u ---
    # Wartości brzegowe pola u na węzłach znanych: przeszkody -> u = x - c_i,
    # zewnętrzne ściany -> 0 (pozostają zerowe z inicjalizacji).
    val_ux_full = np.zeros((ny, nx), dtype=np.float64)
    val_uy_full = np.zeros((ny, nx), dtype=np.float64)
    if objects_list:
        cx_arr = np.array(
            [centers[i][0] for i in range(len(objects_list))]
        )
        cy_arr = np.array(
            [centers[i][1] for i in range(len(objects_list))]
        )
        obj_ids = obstacle_id_map[mask]
        ux = X[mask] - cx_arr[obj_ids]
        uy = Y[mask] - cy_arr[obj_ids]

        vecs = np.column_stack([ux, uy])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)

        ux_norm = ux / safe_norms[:, 0]
        uy_norm = uy / safe_norms[:, 0]

        val_ux_full[mask] = ux_norm
        val_uy_full[mask] = uy_norm

    # Rozwiązujemy niezależnie dla składowych u_x oraz u_y (ten sam wkład warunków brzegowych)
    rhs_ux = -(A_bc @ val_ux_full.reshape(-1)[known_flat])
    rhs_uy = -(A_bc @ val_uy_full.reshape(-1)[known_flat])

    # Jedna faktoryzacja LU macierzy A, wykorzystywana wielokrotnie (u_x, u_y, h)
    lu = splu(A)
    u_sol = lu.solve(np.column_stack([rhs_ux, rhs_uy]))
    u_x_sol = u_sol[:, 0]
    u_y_sol = u_sol[:, 1]

    # Odtwarzanie pełnych pól składowych u
    u_x = np.zeros((ny, nx), dtype=np.float32)
    u_y = np.zeros((ny, nx), dtype=np.float32)
    u_x.reshape(-1)[unknown_flat] = u_x_sol
    u_y.reshape(-1)[unknown_flat] = u_y_sol
    u_x.reshape(-1)[known_flat] = val_ux_full.reshape(-1)[known_flat]
    u_y.reshape(-1)[known_flat] = val_uy_full.reshape(-1)[known_flat]

    # Numeryczne obliczanie gradientów pola u
    duy_dy, duy_dx = np.gradient(u_y, y, x)
    dux_dy, dux_dx = np.gradient(u_x, y, x)

    # Wyznaczenie modułu gradientu (człon źródłowy równania Poissona): f = ||grad u||
    norm_grad_u = np.sqrt(
        dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2 + 1e-8
    )

    # --- 2. ROZWIĄZYWANIE RÓWNANIA POISSONA DLA FUNKCJI BEZPIECZEŃSTWA h ---
    # Równanie: Delta h = -||grad u||  =>  A * h = -norm_grad_u
    # (h = 0 na wszystkich węzłach znanych, więc wkład warunków brzegowych do RHS jest zerowy)
    rhs_h = -norm_grad_u.reshape(-1)[unknown_flat].astype(np.float64)
    h_sol = lu.solve(rhs_h)

    # Odtwarzanie pełnego pola h
    h = np.zeros((ny, nx), dtype=np.float32)
    h.reshape(-1)[unknown_flat] = h_sol

    # Obliczanie pochodnych przestrzennych funkcji h
    dhdy, dhdx = np.gradient(h, y, x)

    # Czyszczenie i filtracja wartości nieliczbowych (NaN)
    h = np.nan_to_num(h, nan=0.0)
    dhdx = np.nan_to_num(dhdx, nan=0.0)
    dhdy = np.nan_to_num(dhdy, nan=0.0)
    grid = mask.astype(np.uint8)

    return u_x, u_y, h, dhdx, dhdy, grid


def make_objects_list_for_index(
    map_index,
    max_objects_number=12,
):
    """Generuje losowe rozłożenie przeszkód wewnątrz obszaru [-5, 5]^2.
    Przeszkody mogą być kołowe lub prostokątne.
    """
    rng = np.random.RandomState(map_index)
    objects_number = int(math.ceil(max_objects_number * rng.rand()))
    objects_list = []

    if map_index % 20 == 0:
        return objects_list

    def object_distance(a, b):
        type_a = a.get("type", "circle")
        type_b = b.get("type", "circle")
        cx_a, cy_a = float(a["c"][0]), float(a["c"][1])
        cx_b, cy_b = float(b["c"][0]), float(b["c"][1])

        if type_a in ("rect", "rectangle"):
            hx_a, hy_a = float(a["size"][0]), float(a["size"][1])
        else:
            r_a = float(a["r"])

        if type_b in ("rect", "rectangle"):
            hx_b, hy_b = float(b["size"][0]), float(b["size"][1])
        else:
            r_b = float(b["r"])

        if type_a not in ("rect", "rectangle") and type_b not in (
            "rect",
            "rectangle",
        ):
            return np.linalg.norm([cx_a - cx_b, cy_a - cy_b]) - (
                r_a + r_b
            )

        if type_a in ("rect", "rectangle") and type_b in (
            "rect",
            "rectangle",
        ):
            dx = max(0.0, abs(cx_a - cx_b) - (hx_a + hx_b))
            dy = max(0.0, abs(cy_a - cy_b) - (hy_a + hy_b))
            return np.hypot(dx, dy)

        if type_a in ("rect", "rectangle"):
            dx = max(0.0, abs(cx_b - cx_a) - hx_a)
            dy = max(0.0, abs(cy_b - cy_a) - hy_a)
            return np.hypot(dx, dy) - r_b

        dx = max(0.0, abs(cx_a - cx_b) - hx_b)
        dy = max(0.0, abs(cy_a - cy_b) - hy_b)
        return np.hypot(dx, dy) - r_a

    for _ in range(objects_number):
        attempts = 0
        max_attempts = 200
        valid = False
        while not valid and attempts < max_attempts:
            attempts += 1
            shape_type = "rect" if rng.rand() < 0.45 else "circle"

            if shape_type == "rect":
                half_w = rng.uniform(0.2, 0.55)
                half_h = rng.uniform(0.2, 0.55)
                min_xy = np.array([-4.3 + half_w, -4.3 + half_h])
                max_xy = np.array([4.3 - half_w, 4.3 - half_h])
                potential_c = rng.uniform(min_xy, max_xy)
                potential_obj = {
                    "type": "rect",
                    "c": potential_c.copy(),
                    "size": np.array([half_w, half_h], dtype=float),
                }
            else:
                potential_r = rng.uniform(0.25, 0.5)
                potential_c = rng.uniform(
                    -4.3 + potential_r, 4.3 - potential_r, 2
                )
                potential_obj = {
                    "type": "circle",
                    "c": potential_c.copy(),
                    "r": float(potential_r),
                }

            overlap = False
            for obj in objects_list:
                if object_distance(potential_obj, obj) < 0.4:
                    overlap = True
                    break

            if not overlap:
                valid = True
                objects_list.append(potential_obj)

    return objects_list


def compute_map_result(
    map_index,
    res,
):
    objects_list = make_objects_list_for_index(map_index)
    u_x, u_y, h, dhdx, dhdy, grid = solve_laplace_and_poisson(
        objects_list, n=res
    )
    return map_index, u_x, u_y, h, dhdx, dhdy, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name",
        nargs="?",
        default="nik_training_data_512x512_50.h5",
    )
    parser.add_argument(
        "generated_maps_number", nargs="?", type=int, default=50
    )
    parser.add_argument(
        "resolution", nargs="?", type=int, default=512
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Liczba wątków do równoległych obliczeń map (min 1)",
    )
    args = parser.parse_args()

    file_name = args.file_name
    generated_maps_number = args.generated_maps_number
    res = args.resolution
    num_threads = max(1, int(args.num_threads))

    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    map_range = range(1, generated_maps_number + 1)
    compute_fn = partial(compute_map_result, res=res)

    if num_threads == 1:
        results_iter = map(compute_fn, map_range)
    else:
        # Obliczenia są równoległe, ale zapis do HDF5 pozostaje sekwencyjny dla bezpieczeństwa.
        executor = ThreadPoolExecutor(max_workers=num_threads)
        results_iter = executor.map(compute_fn, map_range)

    try:
        for map_index, u_x, u_y, h, dhdx, dhdy, grid in tqdm(
            results_iter,
            total=generated_maps_number,
            desc=f"Generating maps ({num_threads} threads)",
        ):
            index_string = f"{map_index:06d}"
            with h5py.File(file_name, "a") as f:
                # Zapisujemy macierze o kształcie [1, H, W] zgodnym z architekturami splotowymi PyTorch
                grid_data = np.expand_dims(grid, axis=0).astype(
                    np.uint8
                )
                h_data = np.expand_dims(h, axis=0).astype(np.float32)
                u_x_data = np.expand_dims(u_x, axis=0).astype(
                    np.float32
                )
                u_y_data = np.expand_dims(u_y, axis=0).astype(
                    np.float32
                )
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
    finally:
        if num_threads > 1:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
