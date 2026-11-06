"""Convert occupancy grids from an H5 file into ROS2 map_server files (.pgm + .yaml).

Input H5 layout: /grid/<key> -> 2D array, 1 = occupied, 0 = free (see load_grids_for_inference).
Output (per sample): <out-dir>/<prefix>_<key>.pgm and .yaml, same format as map_server maps.
"""
import argparse
import os

import h5py
import numpy as np

DEFAULT_H5 = "data/Carla/carla_10_08_examples_grid.h5"

OCCUPIED_PX = 0     # black
FREE_PX = 254       # map_server "free" shade used by SLAM-generated maps


def write_pgm(path, img):
    h, w = img.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(img.astype(np.uint8).tobytes())


def write_yaml(path, image_name, resolution, origin):
    with open(path, "w") as f:
        f.write(
            f"image: {image_name}\n"
            f"mode: trinary\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin[0]}, {origin[1]}, 0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.25\n"
        )


def grid_to_image(grid, invert=False, flip_y=True, transpose=False):
    grid = np.asarray(grid)
    if grid.ndim == 3:
        grid = grid[0]
    if transpose:
        grid = grid.T
    occupied = grid > 0
    if invert:
        occupied = ~occupied
    img = np.where(occupied, OCCUPIED_PX, FREE_PX).astype(np.uint8)
    # PGM row 0 is the top of the map (max y); flip if grid row 0 is min y.
    return img[::-1] if flip_y else img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=DEFAULT_H5)
    p.add_argument("--out-dir", default="data/Carla/ros_maps")
    p.add_argument("--prefix", default="map")
    p.add_argument("--sample-index", type=int, default=None,
                   help="Convert only this sample. Omit for all.")
    p.add_argument("--resolution", type=float, default=0.05, help="Meters per cell.")
    p.add_argument("--center-origin", action="store_true", default=True,
                   help="Place the map origin so the grid is centered on (0,0) (default).")
    p.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="Explicit lower-left origin; overrides centering.")
    p.add_argument("--invert", action="store_true",
                   help="Treat 0 as occupied and 1 as free.")
    p.add_argument("--no-flip-y", action="store_true",
                   help="Do not flip rows (use if grid row 0 is already the top).")
    p.add_argument("--transpose", action="store_true", help="Transpose grid before export.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.data_path, "r") as h:
        keys = sorted(h["grid"].keys())
        if args.sample_index is not None:
            keys = [keys[args.sample_index]]
        for key in keys:
            img = grid_to_image(h["grid"][key][()], args.invert,
                                not args.no_flip_y, args.transpose)
            hgt, wid = img.shape
            if args.origin is not None:
                origin = args.origin
            else:
                origin = (round(-wid * args.resolution / 2, 3),
                          round(-hgt * args.resolution / 2, 3))
            name = f"{args.prefix}_{key}"
            write_pgm(os.path.join(args.out_dir, name + ".pgm"), img)
            write_yaml(os.path.join(args.out_dir, name + ".yaml"),
                       name + ".pgm", args.resolution, origin)
            print(f"{name}: {wid}x{hgt}, origin={list(origin)}")


if __name__ == "__main__":
    main()
