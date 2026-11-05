#!/usr/bin/env python3
"""Measure trajectory distance to obstacle strokes extracted from a plot image."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


DEFAULT_PLOT_BOUNDS = (99.0, 1178.0, 53.0, 658.0)
DEFAULT_WORLD_BOUNDS = (-1.2, 16.8, -5.0, 5.1)
SCENARIO_CALIBRATIONS = {
    "carla_08": {
        "plot_bounds": (109.0, 1178.0, 86.0, 1155.0),
        "world_bounds": (-12.8, 12.8, -12.8, 12.8),
    },
}
EXPECTED_START = (-10.0, 5.0)
EXPECTED_GOAL = (5.0, 10.0)
PROGRESS_SAMPLES = 101
MIN_OBSTACLE_COMPONENT_PIXELS = 50


def _read_trajectories(csv_path: Path) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, float, int, int], list[dict[str, float]]
    ] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"scenario", "c", "rep", "seed", "idx", "x", "y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                key = (
                    row["scenario"],
                    float(row["c"]),
                    int(row["rep"]),
                    int(row["seed"]),
                )
                point = {
                    "idx": int(row["idx"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid trajectory data on CSV line {line_number}"
                ) from error
            grouped[key].append(point)

    trajectories = []
    for key, points in sorted(grouped.items()):
        points.sort(key=lambda point: point["idx"])
        indices = [point["idx"] for point in points]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"Duplicate node index in trajectory {key}"
            )
        coordinates = np.array(
            [[point["x"], point["y"]] for point in points],
            dtype=float,
        )
        if len(coordinates) < 2 or not np.isfinite(coordinates).all():
            raise ValueError(
                f"Trajectory {key} must have at least two finite nodes"
            )
        for actual, expected, label in (
            (coordinates[0], EXPECTED_START, "start"),
            (coordinates[-1], EXPECTED_GOAL, "goal"),
        ):
            if not np.allclose(actual, expected, atol=1e-3, rtol=0.0):
                raise ValueError(
                    f"Trajectory {key} has unexpected {label}: "
                    f"({actual[0]:.4f}, {actual[1]:.4f})"
                )
        trajectories.append({"key": key, "points": coordinates})

    if not trajectories:
        raise ValueError(f"No trajectories found in {csv_path}")
    return trajectories


def _remove_small_components(
    mask: np.ndarray, *, min_pixels: int
) -> np.ndarray:
    labels, _ = label(mask)
    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def _obstacle_distance_field(
    image_path: Path,
    plot_bounds: tuple[float, float, float, float],
    world_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    left, right, top, bottom = plot_bounds
    x_min, x_max, y_min, y_max = world_bounds
    if not (
        0 <= left < right < image.shape[1]
        and 0 <= top < bottom < image.shape[0]
    ):
        raise ValueError(
            "Plot pixel bounds must lie inside the image"
        )
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("World bounds must be increasing")

    # Ignore border lines, then retain sizable neutral-gray obstacle regions.
    margin = 8
    x0, x1 = (
        int(np.ceil(left + margin)),
        int(np.floor(right - margin)) + 1,
    )
    y0, y1 = (
        int(np.ceil(top + margin)),
        int(np.floor(bottom - margin)) + 1,
    )
    plot_pixels = image[y0:y1, x0:x1].astype(np.int16)
    dark_pixels = (
        (plot_pixels.min(axis=2) > 35)
        & (plot_pixels.max(axis=2) < 110)
        & ((plot_pixels.max(axis=2) - plot_pixels.min(axis=2)) <= 5)
    )
    obstacle_pixels = _remove_small_components(
        dark_pixels, min_pixels=MIN_OBSTACLE_COMPONENT_PIXELS
    )
    if not obstacle_pixels.any():
        raise ValueError(
            "No obstacle strokes detected; check image and plot calibration"
        )

    # EDT gives distance to the nearest obstacle pixel, converted from pixels to meters.
    pixel_width = (x_max - x_min) / (right - left)
    pixel_height = (y_max - y_min) / (bottom - top)
    distance_m = distance_transform_edt(
        ~obstacle_pixels, sampling=(pixel_height, pixel_width)
    )
    return distance_m, (x0, x1, y0, y1)


def _resample_polyline(
    points: np.ndarray, progress: np.ndarray
) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 0))
    points = points[keep]
    if len(points) < 2:
        raise ValueError(
            "A trajectory must contain at least two distinct nodes"
        )
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(np.diff(points, axis=0), axis=1)
            ),
        )
    )
    targets = progress * cumulative[-1]
    return np.column_stack(
        (
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        )
    )


def _sample_distance_field(
    points: np.ndarray,
    distance_field: np.ndarray,
    crop: tuple[int, int, int, int],
    plot_bounds: tuple[float, float, float, float],
    world_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x0, x1, y0, y1 = crop
    left, right, top, bottom = plot_bounds
    x_min, x_max, y_min, y_max = world_bounds
    pixel_x = left + (points[:, 0] - x_min) * (right - left) / (
        x_max - x_min
    )
    pixel_y = top + (y_max - points[:, 1]) * (bottom - top) / (
        y_max - y_min
    )
    col = np.rint(pixel_x).astype(int) - x0
    row = np.rint(pixel_y).astype(int) - y0
    if np.any(
        (col < 0) | (col >= x1 - x0) | (row < 0) | (row >= y1 - y0)
    ):
        raise ValueError(
            "A sampled trajectory point falls outside the calibrated plot area"
        )
    return distance_field[row, col]


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    path: Path, summary: list[dict[str, object]]
) -> None:
    values_by_c: dict[float, list[dict[str, object]]] = defaultdict(
        list
    )
    for row in summary:
        values_by_c[float(row["c"])].append(row)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for c, rows in sorted(values_by_c.items()):
        rows.sort(key=lambda row: float(row["progress_percent"]))
        progress = np.array(
            [float(row["progress_percent"]) for row in rows]
        )
        mean = np.array(
            [float(row["mean_distance_m"]) for row in rows]
        )
        std = np.array([float(row["std_distance_m"]) for row in rows])
        count = int(rows[0]["n_trajectories"])
        label = f"c = {c:g} ({count} trajectories)"
        (line,) = ax.plot(progress, mean, label=label)
        ax.fill_between(
            progress,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.2,
        )

    ax.set(
        xlabel="Normalized trajectory length (%)",
        ylabel="Distance to nearest obstacle (m)",
        title="Mean obstacle distance along trajectories",
        xlim=(0, 100),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure trajectory distance to the dark obstacle strokes in a plot image."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Trajectory CSV containing scenario, c, rep, idx, x, and y",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Plot image showing the obstacle strokes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV results and plot (default: next to the input CSV)",
    )
    parser.add_argument(
        "--plot-bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        help="Plot frame pixel bounds; defaults are calibrated for the supplied corridor_zigzag_3 image",
    )
    parser.add_argument(
        "--world-bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="World-coordinate bounds corresponding to the plot frame",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir or args.csv.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _read_trajectories(args.csv)
    scenarios = {trajectory["key"][0] for trajectory in trajectories}
    calibration = (
        SCENARIO_CALIBRATIONS.get(next(iter(scenarios)), {})
        if len(scenarios) == 1
        else {}
    )
    plot_bounds = tuple(
        args.plot_bounds
        if args.plot_bounds is not None
        else calibration.get("plot_bounds", DEFAULT_PLOT_BOUNDS)
    )
    world_bounds = tuple(
        args.world_bounds
        if args.world_bounds is not None
        else calibration.get("world_bounds", DEFAULT_WORLD_BOUNDS)
    )
    distance_field, crop = _obstacle_distance_field(
        args.image, plot_bounds, world_bounds
    )

    progress = np.linspace(0.0, 1.0, PROGRESS_SAMPLES)
    profile_rows: list[dict[str, object]] = []
    path_summary: list[dict[str, object]] = []
    samples_by_c: dict[float, list[np.ndarray]] = defaultdict(list)
    for trajectory in trajectories:
        scenario, c, rep, seed = trajectory["key"]
        sampled_points = _resample_polyline(
            trajectory["points"], progress
        )
        distances = _sample_distance_field(
            sampled_points,
            distance_field,
            crop,
            plot_bounds,
            world_bounds,
        )
        samples_by_c[c].append(distances)
        for percent, distance in zip(progress * 100.0, distances):
            profile_rows.append(
                {
                    "scenario": scenario,
                    "c": c,
                    "rep": rep,
                    "seed": seed,
                    "progress_percent": f"{percent:.1f}",
                    "distance_to_obstacle_m": f"{distance:.6f}",
                }
            )
        path_summary.append(
            {
                "scenario": scenario,
                "c": c,
                "rep": rep,
                "seed": seed,
                "mean_distance_m": f"{np.mean(distances):.6f}",
                "std_distance_along_path_m": f"{np.std(distances, ddof=1):.6f}",
                "minimum_distance_m": f"{np.min(distances):.6f}",
            }
        )

    summary_rows: list[dict[str, object]] = []
    for c, path_profiles in sorted(samples_by_c.items()):
        stacked = np.stack(path_profiles)
        means = stacked.mean(axis=0)
        stds = (
            stacked.std(axis=0, ddof=1)
            if len(stacked) > 1
            else np.zeros_like(means)
        )
        for percent, mean, std in zip(progress * 100.0, means, stds):
            summary_rows.append(
                {
                    "c": c,
                    "progress_percent": f"{percent:.1f}",
                    "mean_distance_m": f"{mean:.6f}",
                    "std_distance_m": f"{std:.6f}",
                    "n_trajectories": len(stacked),
                }
            )

    profile_path = output_dir / "trajectory_distance_profiles.csv"
    summary_path = output_dir / "trajectory_distance_summary.csv"
    path_summary_path = output_dir / "trajectory_distance_by_path.csv"
    plot_path = output_dir / "trajectory_distance_plot.png"
    _write_csv(
        profile_path,
        [
            "scenario",
            "c",
            "rep",
            "seed",
            "progress_percent",
            "distance_to_obstacle_m",
        ],
        profile_rows,
    )
    _write_csv(
        summary_path,
        [
            "c",
            "progress_percent",
            "mean_distance_m",
            "std_distance_m",
            "n_trajectories",
        ],
        summary_rows,
    )
    _write_csv(
        path_summary_path,
        [
            "scenario",
            "c",
            "rep",
            "seed",
            "mean_distance_m",
            "std_distance_along_path_m",
            "minimum_distance_m",
        ],
        path_summary,
    )
    _plot_summary(plot_path, summary_rows)

    print(f"Analyzed {len(trajectories)} trajectories.")
    print(f"Per-trajectory profiles: {profile_path}")
    print(f"Mean and standard deviation by progress: {summary_path}")
    print(f"Per-trajectory distance statistics: {path_summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
