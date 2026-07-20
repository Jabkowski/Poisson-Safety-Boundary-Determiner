import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_PATH = "nik_training_data_128x128_5.h5"

DEFAULT_EXTENT = (-5.0, 5.0, -5.0, 5.0)


def _to_2d(value):
    array = np.asarray(value)
    array = np.squeeze(array)

    while array.ndim > 2:
        array = array[0]

    return array


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


def plot_poisson_reference_example(
    grid,
    h_true,
    ux_true,
    uy_true,
    save_path=None,
    show=False,
    extent=DEFAULT_EXTENT,
):
    """Plot the real vector field and the reference safety function for one sample."""
    grid = _to_2d(grid)
    h_true = _to_2d(h_true)
    ux_true = _to_2d(ux_true)
    uy_true = _to_2d(uy_true)

    if grid.shape != h_true.shape:
        raise ValueError(
            f"grid and h_true must have the same shape, got {grid.shape} and {h_true.shape}"
        )

    x_min, x_max, y_min, y_max = extent
    x = np.linspace(x_min, x_max, grid.shape[1])
    y = np.linspace(y_min, y_max, grid.shape[0])
    X, Y = np.meshgrid(x, y)

    mask = grid == 0
    magnitude = np.sqrt(ux_true**2 + uy_true**2)

    ux_plot = np.where(mask, ux_true, np.nan)
    uy_plot = np.where(mask, uy_true, np.nan)
    h_plot = np.where(mask, h_true, np.nan)
    magnitude = np.where(mask, magnitude, np.nan)

    if ux_plot.shape != X.shape and ux_plot.T.shape == X.shape:
        ux_plot = ux_plot.T
        uy_plot = uy_plot.T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    contour = ax.contourf(X, Y, magnitude, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Magnituda ||u||")
    ax.streamplot(
        X,
        Y,
        ux_plot,
        uy_plot,
        color="white",
        linewidth=0.8,
        density=1.0,
    )
    ax.imshow(
        grid,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="gray_r",
        alpha=0.3,
    )
    ax.set_title("Zharmonizowane pole odpychania $\\mathbf{u}$\n(Streamlines & Magnituda)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")

    ax = axes[1]
    contour = ax.contourf(X, Y, h_plot, levels=50, cmap="plasma")
    fig.colorbar(contour, ax=ax, label="Wartość h")
    ax.imshow(
        grid,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="gray_r",
        alpha=0.3,
    )
    ax.set_title("Referencyjna funkcja bezpieczeństwa\n(Rozwiązanie numeryczne z H5)")
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
        plot_poisson_reference_example(
            grid,
            h_true,
            ux_true,
            uy_true,
            save_path=save_path,
            show=args.show,
        )
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
