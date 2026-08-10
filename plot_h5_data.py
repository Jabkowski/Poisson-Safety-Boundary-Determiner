import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from py_utils.utils import plot_poisson_pinn_example, _to_2d

DEFAULT_DATA_PATH = "data/nik_training_data_512x512_10.h5"

DEFAULT_EXTENT = (-5.0, 5.0, -5.0, 5.0)


def _read_samples(file_path, sample_indices=None):
    """Read one or all samples. Returns list of (grid, h_true, ux_true, uy_true, key)."""
    with h5py.File(file_path, "r") as handle:
        keys = sorted(handle["grid"].keys())
        if not keys:
            raise ValueError(f"No samples found in {file_path}")

        selected = [keys[i] for i in sample_indices] if sample_indices is not None else keys

        samples = []
        for key in selected:
            grid    = _to_2d(handle["grid"][key][()])
            h_true  = _to_2d(handle["h"][key][()])
            ux_true = _to_2d(handle["u_x"][key][()])
            uy_true = _to_2d(handle["u_y"][key][()])
            samples.append((grid, h_true, ux_true, uy_true, key))

    return samples


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot the first H5 sample with the same style used in training, but without model predictions."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Index of sample to plot. Omit to plot all samples.")
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()

    indices = [args.sample_index] if args.sample_index is not None else None
    samples = _read_samples(args.data_path, sample_indices=indices)

    for grid, h_true, ux_true, uy_true, sample_key in samples:
        tmp_path = args.data_path.replace(".h5", f"_sample_{sample_key}.png")
        save_path = os.path.join("fig", os.path.split(tmp_path)[-1].replace("data", "fig"))
        print(f"Plotting sample {sample_key} from {args.data_path}")
        plot_poisson_pinn_example(
            grid,
            h_true,
            ux_true,
            uy_true,
            save_path=save_path,
            show=args.show,
            extent=DEFAULT_EXTENT,
            label="Referencyjna",
        )
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
