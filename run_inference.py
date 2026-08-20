import argparse
import os
import time

from py_utils.utils import plot_poisson_pinn_example

DEFAULT_DATA_PATH = "data/Carla/carla_10_08_examples_grid.h5"
DEFAULT_WEIGHTS_PATH = (
    "weights/poisson_unet_bilinear_model_512x512_2000_e35_ok_inf.pth"
    #poisson_unet_bilinear_model_512x512_2000_e35_ok_inf
    #poisson_first_bilinear_second_UNet_model_512x512_1000_e30_mse_1
)
DEFAULT_EXTENT = (-5.0, 5.0, -5.0, 5.0)


def load_grids_for_inference(h5_path):
    import h5py
    import numpy as np
    import torch

    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle["grid"].keys())
        grids = []

        for key in keys:
            grid = np.asarray(handle["grid"][key][()], dtype=np.float32)
            if grid.ndim == 2:
                grid = grid[None, ...]
            elif grid.ndim != 3:
                raise ValueError(
                    f"Unsupported grid shape for sample {key}: {grid.shape}"
                )
            grids.append(grid)

    return torch.from_numpy(np.stack(grids)).float()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run UNetDoublePoisson inference on H5 grids and save PNG plots."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Index of sample to plot. Omit to run inference on all samples.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference to run on CPU even when CUDA is available.",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()

    import torch

    from train_pinn import UNetDoublePoisson

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if not os.path.isfile(args.weights_path):
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")

    model = UNetDoublePoisson().to(device)
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {total_parameters}")

    grids = load_grids_for_inference(args.data_path)

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= grids.shape[0]:
            raise IndexError(
                f"sample-index {args.sample_index} is out of range for {grids.shape[0]} samples"
            )
        selected = [(args.sample_index, grids[args.sample_index : args.sample_index + 1])]
    else:
        selected = [(index, grids[index : index + 1]) for index in range(grids.shape[0])]

    base_name = os.path.splitext(os.path.basename(args.data_path))[0]

    prediction_seconds = 0.0
    with torch.no_grad():
        for sample_index, grid in selected:
            input_grid = grid.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prediction_start = time.perf_counter()
            pred_3ch = model(input_grid)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"pred: {time.perf_counter() - prediction_start}")
            pred_3ch = pred_3ch.cpu().numpy().squeeze(0)
            ux_pred = pred_3ch[0]
            uy_pred = pred_3ch[1]
            h_pred = pred_3ch[2]
            grid_np = grid.cpu().numpy().squeeze(0)

            save_path = os.path.join(
                "fig",
                f"carla_10_08_grid_{base_name}_sample_{sample_index:06d}_prediction.png",
            )

            print(f"Plotting prediction for sample {sample_index} from {args.data_path}")
            plot_poisson_pinn_example(
                grid_np,
                h_pred,
                ux_pred,
                uy_pred,
                save_path=save_path,
                show=args.show,
                extent=DEFAULT_EXTENT,
                label="Predykcja",
            )
            print(f"Saved plot to {save_path}")
            print(f"Model prediction time: {prediction_seconds:.4f} s")


if __name__ == "__main__":
    main()