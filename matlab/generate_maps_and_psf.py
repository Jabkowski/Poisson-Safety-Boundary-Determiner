#!/usr/bin/env python3
"""
Python translation of matlab/generate_maps_and_psf.m

This script generates maps with random circular obstacles and computes
an approximate Poisson-based safety function `h` together with its
spatial derivatives `dhdx`, `dhdy` on a 512x512 grid.

Usage:
    python matlab/generate_maps_and_psf.py [output_file.h5] [num_maps]

Notes:
- The MATLAB PDE toolbox is approximated here with a finite-difference
  Poisson solver on a regular grid using SciPy sparse linear algebra.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import numpy as np
import h5py
from scipy import sparse
from scipy.sparse.linalg import spsolve


def generate_poisson_safety_function(objects_list, n=512):
    """Generate h, dhdx, dhdy, grid on a unit square [0,1]^2.

    - objects_list: list of dicts with keys 'c' (2-vector) and 'r' (float)
    - returns: h (n,n), dhdx (n,n), dhdy (n,n), grid (n,n uint8)
    """
    # grid coords
    nx = ny = n
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    # mask for obstacle interiors (Dirichlet nodes)
    mask = np.zeros((ny, nx), dtype=bool)
    for obj in objects_list:
        cx, cy = float(obj["c"][0]), float(obj["c"][1])
        r = float(obj["r"])
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask |= dist <= r + 1e-12

    # Right-hand side f: we approximate with 1.0 outside obstacles, 0 inside
    f = np.ones((ny, nx), dtype=np.float32)
    f[mask] = 0.0

    # Unknown nodes are those not on outer boundary and not in obstacle interiors
    unknown = np.ones((ny, nx), dtype=bool)
    unknown[0, :] = False
    unknown[-1, :] = False
    unknown[:, 0] = False
    unknown[:, -1] = False
    unknown[mask] = False

    idx = -np.ones((ny, nx), dtype=int)
    idx_flat = 0
    for j in range(ny):
        for i in range(nx):
            if unknown[j, i]:
                idx[j, i] = idx_flat
                idx_flat += 1

    N = idx_flat
    if N == 0:
        # trivial empty domain
        h = np.zeros((ny, nx), dtype=np.float32)
        dhdx = np.zeros_like(h)
        dhdy = np.zeros_like(h)
        grid = (h > 0).astype(np.uint8)
        return h, dhdx, dhdy, grid

    data = []
    rows = []
    cols = []
    rhs = np.zeros(N, dtype=np.float64)

    def add_entry(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    # 5-point Laplacian: (u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j})/dx^2 = -f
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not unknown[j, i]:
                continue
            row = idx[j, i]
            # center
            add_entry(row, row, -4.0 / (dx * dx))

            # neighbors
            for jj, ii in (
                (j - 1, i),
                (j + 1, i),
                (j, i - 1),
                (j, i + 1),
            ):
                if unknown[jj, ii]:
                    add_entry(row, idx[jj, ii], 1.0 / (dx * dx))
                else:
                    # Dirichlet known value (zero) contributes nothing to RHS
                    pass

            rhs[row] = -float(f[j, i])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Solve linear system A * h_unknown = rhs
    h_unknown = spsolve(A, rhs)

    # assemble full h
    h = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            if unknown[j, i]:
                h[j, i] = h_unknown[idx[j, i]]
            else:
                h[j, i] = 0.0

    # compute gradients (note: numpy.gradient expects axis order)
    dhdy, dhdx = np.gradient(h, y, x)

    # remove NaNs
    h = np.nan_to_num(h, nan=0.0)
    dhdx = np.nan_to_num(dhdx, nan=0.0)
    dhdy = np.nan_to_num(dhdy, nan=0.0)

    grid = (h > 0).astype(np.uint8)
    return h, dhdx, dhdy, grid


def make_objects_list_for_index(map_index, max_objects_number=50):
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
            potential_c = rng.rand(2)
            potential_r = 0.1 * rng.rand() + 0.02
            margin = 1e-4
            if (
                (potential_c[0] - potential_r < margin)
                or (potential_c[0] + potential_r > 1 - margin)
                or (potential_c[1] - potential_r < margin)
                or (potential_c[1] + potential_r > 1 - margin)
            ):
                continue

            overlap = False
            for obj in objects_list:
                dist_centers = np.linalg.norm(
                    potential_c - np.array(obj["c"])
                )
                if dist_centers < (potential_r + obj["r"] + 0.005):
                    overlap = True
                    break

            if not overlap:
                valid = True
                objects_list.append(
                    {"c": potential_c.copy(), "r": float(potential_r)}
                )

        if attempts >= max_attempts:
            print(
                f"Warning: map {map_index}: reached packing attempts limit, generated {len(objects_list)} of {objects_number}"
            )
            break

    return objects_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name", nargs="?", default="training_data_512x512.h5"
    )
    parser.add_argument(
        "generated_maps_number", nargs="?", type=int, default=10
    )
    args = parser.parse_args()

    file_name = args.file_name
    generated_maps_number = args.generated_maps_number

    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

    for map_index in range(1, generated_maps_number + 1):
        objects_list = make_objects_list_for_index(map_index)

        h, dhdx, dhdy, grid = generate_poisson_safety_function(
            objects_list
        )

        index_string = f"{map_index:06d}"
        with h5py.File(file_name, "a") as f:
            grp_grid = f.require_group("grid")
            dset_name = index_string
            if dset_name in grp_grid:
                del grp_grid[dset_name]
            grp_grid.create_dataset(
                dset_name, data=grid.astype(np.uint8), dtype="u1"
            )

            grp_h = f.require_group("h")
            if dset_name in grp_h:
                del grp_h[dset_name]
            grp_h.create_dataset(
                dset_name, data=h.astype(np.float32), dtype="f4"
            )

            grp_dhdx = f.require_group("dhdx")
            if dset_name in grp_dhdx:
                del grp_dhdx[dset_name]
            grp_dhdx.create_dataset(
                dset_name, data=dhdx.astype(np.float32), dtype="f4"
            )

            grp_dhdy = f.require_group("dhdy")
            if dset_name in grp_dhdy:
                del grp_dhdy[dset_name]
            grp_dhdy.create_dataset(
                dset_name, data=dhdy.astype(np.float32), dtype="f4"
            )

        print(
            f"Generated map {map_index}/{generated_maps_number} -> {file_name}:{index_string}"
        )


if __name__ == "__main__":
    main()
