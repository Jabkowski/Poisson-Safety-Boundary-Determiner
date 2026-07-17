#!/usr/bin/env python3
"""
Skrypt do generowania map przeszkód i obliczania funkcji bezpieczeństwa Poissona (PSF)
przy użyciu solvera różnic skończonych na regularnej siatce.

Domyślnie uruchamia się w trybie MATLAB-compatible:
- domena [0.0, 1.0]^2,
- losowanie przeszkód zgodne z generate_maps_and_psf.m,
- człon źródłowy Poissona f = ||u||.

Domyślna rozdzielczość siatki: 128x128.
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


def solve_laplace_and_poisson(
    objects_list,
    n=128,
    domain_min=0.0,
    domain_max=1.0,
    ux_bc_value=0.01,
    uy_bc_value=0.01,
    source_mode="matlab",
):
    """
    Rozwiązuje sprzężony układ równań na zadanym kwadracie [domain_min, domain_max]^2:
    1) Delta u = 0 w Omega, z warunkami Dirichleta na brzegu
    2) Delta h = -f w Omega, h = 0 na krawędziach przeszkód i ścianach zewnętrznych

    - objects_list: lista słowników z kluczami 'c' (środek) i 'r' (promień)
    - n: rozdzielczość siatki (domyślnie 128)
    - ux_bc_value, uy_bc_value: stałe wartości Dirichleta dla pola u na brzegu
    - source_mode: "matlab" -> f = ||u||, "legacy" -> f = ||grad u||
    - zwraca: h (n,n), dhdx (n,n), dhdy (n,n), grid (n,n uint8)
    """
    nx = ny = n
    x = np.linspace(domain_min, domain_max, nx)
    y = np.linspace(domain_min, domain_max, ny)
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
        [off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr"
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
    # MATLAB-compatible default: stała wartość na całym brzegu (zewnętrznym i przeszkodach).
    val_ux_full = np.zeros((ny, nx), dtype=np.float64)
    val_uy_full = np.zeros((ny, nx), dtype=np.float64)
    val_ux_full.reshape(-1)[known_flat] = float(ux_bc_value)
    val_uy_full.reshape(-1)[known_flat] = float(uy_bc_value)

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

    # Człon źródłowy równania Poissona:
    # - matlab: f = ||u||
    # - legacy: f = ||grad u||
    if source_mode == "matlab":
        source_f = np.sqrt(u_x**2 + u_y**2 + 1e-8)
    elif source_mode == "legacy":
        duy_dy, duy_dx = np.gradient(u_y, y, x)
        dux_dy, dux_dx = np.gradient(u_x, y, x)
        source_f = np.sqrt(
            dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2 + 1e-8
        )
    else:
        raise ValueError(
            "Invalid source_mode. Expected 'matlab' or 'legacy'."
        )

    # --- 2. ROZWIĄZYWANIE RÓWNANIA POISSONA DLA FUNKCJI BEZPIECZEŃSTWA h ---
    # Równanie: Delta h = -f  =>  A * h = -source_f
    # (h = 0 na wszystkich węzłach znanych, więc wkład warunków brzegowych do RHS jest zerowy)
    rhs_h = -source_f.reshape(-1)[unknown_flat].astype(np.float64)
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
    max_objects_number=50,
    domain_min=0.0,
    domain_max=1.0,
    radius_min=0.02,
    radius_max=0.12,
    overlap_margin=0.005,
    boundary_margin=1e-4,
):
    """Generuje losowe rozłożenie przeszkód na domenie kwadratowej."""
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
            # MATLAB-compatible sampling: środek jednostajnie w domenie i promień z przedziału [0.02, 0.12]
            potential_c = rng.uniform(domain_min, domain_max, 2)
            potential_r = rng.uniform(radius_min, radius_max)

            if (
                potential_c[0] - potential_r < domain_min + boundary_margin
                or potential_c[0] + potential_r > domain_max - boundary_margin
                or potential_c[1] - potential_r < domain_min + boundary_margin
                or potential_c[1] + potential_r > domain_max - boundary_margin
            ):
                continue

            overlap = False
            for obj in objects_list:
                dist_centers = np.linalg.norm(
                    potential_c - np.array(obj["c"])
                )
                if dist_centers < (potential_r + obj["r"] + overlap_margin):
                    overlap = True
                    break

            if not overlap:
                valid = True
                objects_list.append(
                    {"c": potential_c.copy(), "r": float(potential_r)}
                )

    return objects_list


def compute_map_result(
    map_index,
    res,
    max_objects_number,
    domain_min,
    domain_max,
    radius_min,
    radius_max,
    overlap_margin,
    boundary_margin,
    ux_bc_value,
    uy_bc_value,
    source_mode,
):
    objects_list = make_objects_list_for_index(
        map_index,
        max_objects_number=max_objects_number,
        domain_min=domain_min,
        domain_max=domain_max,
        radius_min=radius_min,
        radius_max=radius_max,
        overlap_margin=overlap_margin,
        boundary_margin=boundary_margin,
    )
    u_x, u_y, h, dhdx, dhdy, grid = solve_laplace_and_poisson(
        objects_list,
        n=res,
        domain_min=domain_min,
        domain_max=domain_max,
        ux_bc_value=ux_bc_value,
        uy_bc_value=uy_bc_value,
        source_mode=source_mode,
    )
    return map_index, u_x, u_y, h, dhdx, dhdy, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", nargs="?", default="nik_fixed_training_data_512x512_2000.h5"
    )
    parser.add_argument(
        "generated_maps_number", nargs="?", type=int, default=2000
    )
    parser.add_argument(
        "resolution", nargs="?", type=int, default=512
    )
    parser.add_argument(
        "--domain-min",
        type=float,
        default=0.0,
        help="Minimalna współrzędna domeny (MATLAB: 0.0)",
    )
    parser.add_argument(
        "--domain-max",
        type=float,
        default=1.0,
        help="Maksymalna współrzędna domeny (MATLAB: 1.0)",
    )
    parser.add_argument(
        "--max-objects-number",
        type=int,
        default=50,
        help="Maksymalna liczba przeszkód losowanych na mapę (MATLAB: 50)",
    )
    parser.add_argument(
        "--radius-min",
        type=float,
        default=0.02,
        help="Minimalny promień przeszkody (MATLAB: 0.02)",
    )
    parser.add_argument(
        "--radius-max",
        type=float,
        default=0.12,
        help="Maksymalny promień przeszkody (MATLAB: 0.12)",
    )
    parser.add_argument(
        "--overlap-margin",
        type=float,
        default=0.005,
        help="Minimalna separacja między przeszkodami (MATLAB: 0.005)",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=1e-4,
        help="Margines od zewnętrznego brzegu dla centrów przeszkód (MATLAB: 1e-4)",
    )
    parser.add_argument(
        "--ux-bc-value",
        type=float,
        default=0.01,
        help="Dirichlet BC dla składowej u_x na brzegu (MATLAB: 0.01)",
    )
    parser.add_argument(
        "--uy-bc-value",
        type=float,
        default=0.01,
        help="Dirichlet BC dla składowej u_y na brzegu (MATLAB: 0.01)",
    )
    parser.add_argument(
        "--source-mode",
        choices=["matlab", "legacy"],
        default="matlab",
        help='Tryb źródła równania Poissona: "matlab" -> ||u||, "legacy" -> ||grad u||',
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
    domain_min = float(args.domain_min)
    domain_max = float(args.domain_max)
    max_objects_number = max(0, int(args.max_objects_number))
    radius_min = float(args.radius_min)
    radius_max = float(args.radius_max)
    overlap_margin = float(args.overlap_margin)
    boundary_margin = float(args.boundary_margin)
    ux_bc_value = float(args.ux_bc_value)
    uy_bc_value = float(args.uy_bc_value)
    source_mode = args.source_mode

    if domain_max <= domain_min:
        raise ValueError("domain_max must be greater than domain_min")
    if radius_min <= 0.0 or radius_max <= 0.0 or radius_max < radius_min:
        raise ValueError("Invalid radius range")

    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    map_range = range(1, generated_maps_number + 1)
    compute_fn = partial(
        compute_map_result,
        res=res,
        max_objects_number=max_objects_number,
        domain_min=domain_min,
        domain_max=domain_max,
        radius_min=radius_min,
        radius_max=radius_max,
        overlap_margin=overlap_margin,
        boundary_margin=boundary_margin,
        ux_bc_value=ux_bc_value,
        uy_bc_value=uy_bc_value,
        source_mode=source_mode,
    )

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
    finally:
        if num_threads > 1:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
