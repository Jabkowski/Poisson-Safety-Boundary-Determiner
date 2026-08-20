"""Standalone Python port of the ROS2 OccupancyGridMapNode (C++).

ROS / TF dependencies are removed.  The caller is responsible for supplying
world-frame coordinates so the class can be used with any sensor source
(including CARLA radars later).

Coordinate convention (matches the C++ original):
  - X forward, Y left, map_frame = fixed world frame
  - Grid origin = bottom-left corner of the rolling window
  - Occupancy values: 0 (free) .. 100 (fully occupied)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Cell data
# ---------------------------------------------------------------------------

@dataclass
class CellData:
    confidence: float = 0.0
    range_rate_sum: float = 0.0
    range_rate_count: int = 0

    def is_dynamic(self, threshold: float) -> bool:
        if self.range_rate_count == 0:
            return False
        return abs(self.range_rate_sum / self.range_rate_count) > threshold

    def reset(self) -> None:
        self.confidence = 0.0
        self.range_rate_sum = 0.0
        self.range_rate_count = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OccupancyGridMap:
    """Rolling-window occupancy grid with confidence decay and dynamic detection.

    Parameters
    ----------
    map_width_m, map_height_m : float
        Physical extent of the grid [m].
    resolution : float
        Metres per cell.
    decay_rate : float
        Confidence subtracted per second (applied on every ``apply_decay`` call).
    hit_increment : float
        Confidence added per radar detection hit.
    max_confidence : float
        Upper clamp for cell confidence.
    range_rate_threshold : float
        |mean range_rate| above this value marks a cell as dynamic [m/s].
    """

    def __init__(
        self,
        map_width_m: float = 40.0,
        map_height_m: float = 40.0,
        resolution: float = 0.2,
        decay_rate: float = 5.0,
        hit_increment: float = 15.0,
        max_confidence: float = 100.0,
        range_rate_threshold: float = 0.15,
    ) -> None:
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.resolution = resolution
        self.decay_rate = decay_rate
        self.hit_increment = hit_increment
        self.max_confidence = max_confidence
        self.range_rate_threshold = range_rate_threshold

        self.grid_width = math.ceil(map_width_m / resolution)
        self.grid_height = math.ceil(map_height_m / resolution)

        self._grid: List[CellData] = [
            CellData() for _ in range(self.grid_width * self.grid_height)
        ]
        self._lock = threading.Lock()

        self._origin_x: float = 0.0   # bottom-left corner x in world frame
        self._origin_y: float = 0.0   # bottom-left corner y in world frame
        self._origin_initialized: bool = False

        self._last_decay_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_origin(self, robot_x: float, robot_y: float) -> None:
        """Scroll the grid to keep it centred on the robot.

        Safe to call without holding the lock — acquires it internally.
        """
        with self._lock:
            self._update_origin_locked(robot_x, robot_y)

    def add_hit(
        self, world_x: float, world_y: float, range_rate: float = 0.0
    ) -> None:
        """Add a single detection in world-frame coordinates."""
        with self._lock:
            self._accumulate_hit(world_x, world_y, range_rate)

    def add_hits_batch(
        self,
        world_xy: np.ndarray,
        range_rates: Optional[np.ndarray] = None,
    ) -> None:
        """Add a batch of detections.

        Parameters
        ----------
        world_xy : ndarray, shape (N, 2)
            World-frame (x, y) for each detection.
        range_rates : ndarray, shape (N,), optional
            Doppler range-rate for each detection.  Zeros if omitted.
        """
        n = len(world_xy)
        if range_rates is None:
            range_rates = np.zeros(n, dtype=np.float32)

        with self._lock:
            for i in range(n):
                self._accumulate_hit(
                    float(world_xy[i, 0]),
                    float(world_xy[i, 1]),
                    float(range_rates[i]),
                )

    def apply_decay(self, dt: Optional[float] = None) -> None:
        """Subtract decay from all cells.

        Parameters
        ----------
        dt : float, optional
            Elapsed time [s].  If None, derived from wall clock since last call.
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self._last_decay_time
        self._last_decay_time = now

        if dt <= 0.0:
            return

        decay = self.decay_rate * dt

        with self._lock:
            for cell in self._grid:
                if cell.confidence > 0.0:
                    cell.confidence -= decay
                    if cell.confidence <= 0.0:
                        cell.reset()

    @staticmethod
    def compensate_range_rate(
        raw_range_rate: float,
        det_x: float,
        det_y: float,
        robot_x: float,
        robot_y: float,
        robot_vx: float,
        robot_vy: float,
    ) -> float:
        """Remove ego-motion contribution from a measured Doppler range-rate.

        Positive range_rate means the target is moving away from the sensor.
        The robot's radial velocity toward the detection creates a negative
        contribution, so we ADD the radial component to compensate.
        """
        dx = det_x - robot_x
        dy = det_y - robot_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return raw_range_rate
        ux, uy = dx / dist, dy / dist
        v_radial = robot_vx * ux + robot_vy * uy
        return raw_range_rate + v_radial

    # ------------------------------------------------------------------
    # Grid output
    # ------------------------------------------------------------------

    def get_occupancy_array(self) -> np.ndarray:
        """Return a (grid_height, grid_width) int8 array with values 0..100."""
        out = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        with self._lock:
            for gy in range(self.grid_height):
                for gx in range(self.grid_width):
                    c = self._grid[gy * self.grid_width + gx].confidence
                    if c > 0.0:
                        out[gy, gx] = min(
                            100, int(c * 100.0 / self.max_confidence)
                        )
        return out

    def get_colored_bev(
        self,
        img_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """Return an RGB image of the grid for visualisation.

        - Dark background for empty cells.
        - Red  for dynamic cells (|mean range_rate| > threshold).
        - Grey for static occupied cells.
        Alpha is encoded as brightness (brighter = more confident).
        """
        h, w = img_size
        bev = np.full((h, w, 3), 20, dtype=np.uint8)  # dark background

        scale_x = w / self.grid_width
        scale_y = h / self.grid_height

        with self._lock:
            for gy in range(self.grid_height):
                for gx in range(self.grid_width):
                    cell = self._grid[gy * self.grid_width + gx]
                    if cell.confidence <= 0.0:
                        continue

                    alpha = min(1.0, cell.confidence / self.max_confidence)
                    is_dyn = cell.is_dynamic(self.range_rate_threshold)

                    if is_dyn:
                        color = (int(255 * alpha), int(60 * alpha), int(60 * alpha))
                    else:
                        v = int(180 * alpha)
                        color = (v, v, v)

                    # Map grid cell to pixel rect (y-flipped: gy=0 → bottom of image)
                    px0 = int(gx * scale_x)
                    px1 = int((gx + 1) * scale_x)
                    py0 = int((self.grid_height - 1 - gy) * scale_y)
                    py1 = int((self.grid_height - gy) * scale_y)
                    bev[py0:py1, px0:px1] = color

        return bev

    def get_ego_centric_bev(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw_rad: float,
        range_m: float,
        img_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """Ego-centric BEV: vehicle forward = up, centred on ego, rotates with heading.

        Each output pixel is back-projected to its world position via the ego
        pose, then sampled from the occupancy grid.
        Static cells: grey.  Dynamic cells: red.  Ego marker: yellow.
        """
        h, w = img_size
        cx, cy = w // 2, h // 2
        scale = range_m / (min(h, w) / 1.5)  # metres per pixel

        # Pixel coords → vehicle-frame offsets (forward = up = -screen-y)
        px = np.arange(w, dtype=np.float32)
        py = np.arange(h, dtype=np.float32)
        ux, uy = np.meshgrid(px, py)          # shape (h, w)
        x_veh = (cy - uy) * scale             # forward  (positive = ahead)
        y_veh = (ux - cx) * scale             # rightward (positive = right)

        # Vehicle frame → world frame
        cos_e = math.cos(ego_yaw_rad)
        sin_e = math.sin(ego_yaw_rad)
        wx_arr = ego_x + x_veh * cos_e - y_veh * sin_e
        wy_arr = ego_y + x_veh * sin_e + y_veh * cos_e

        bev = np.full((h, w, 3), 20, dtype=np.uint8)

        if not self._origin_initialized:
            return bev

        # World → grid indices
        gx_arr = np.floor((wx_arr - self._origin_x) / self.resolution).astype(np.int32)
        gy_arr = np.floor((wy_arr - self._origin_y) / self.resolution).astype(np.int32)
        valid = (
            (gx_arr >= 0) & (gx_arr < self.grid_width) &
            (gy_arr >= 0) & (gy_arr < self.grid_height)
        )

        # Snapshot grid arrays under lock
        with self._lock:
            conf_flat = np.array([c.confidence    for c in self._grid], dtype=np.float32)
            rrs_flat  = np.array([c.range_rate_sum  for c in self._grid], dtype=np.float32)
            rrc_flat  = np.array([c.range_rate_count for c in self._grid], dtype=np.int32)

        conf_2d = conf_flat.reshape(self.grid_height, self.grid_width)
        rrs_2d  = rrs_flat.reshape(self.grid_height, self.grid_width)
        rrc_2d  = rrc_flat.reshape(self.grid_height, self.grid_width)

        gx_c = np.clip(gx_arr, 0, self.grid_width  - 1)
        gy_c = np.clip(gy_arr, 0, self.grid_height - 1)

        conf   = conf_2d[gy_c, gx_c]
        rrc    = rrc_2d[gy_c, gx_c]
        mean_rr = np.where(rrc > 0, rrs_2d[gy_c, gx_c] / np.maximum(rrc, 1), 0.0)

        occupied = valid & (conf > 0.0)
        dynamic  = occupied & (np.abs(mean_rr) > self.range_rate_threshold)
        static   = occupied & ~dynamic

        alpha = np.clip(conf / self.max_confidence, 0.0, 1.0)

        v = (180.0 * alpha).astype(np.uint8)
        bev[static]  = np.stack([v, v, v], axis=-1)[static]

        r = (255.0 * alpha).astype(np.uint8)
        g = ( 60.0 * alpha).astype(np.uint8)
        bev[dynamic] = np.stack([r, g, g], axis=-1)[dynamic]

        # Yellow ego marker
        bev[cy - 4 : cy + 4, cx - 4 : cx + 4] = (255, 255, 0)
        return bev

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_grid(
        self, wx: float, wy: float
    ) -> Tuple[Optional[int], Optional[int]]:
        """Convert world-frame (x, y) to (gx, gy) grid indices.

        Returns (None, None) if out of bounds or origin not yet initialised.
        """
        if not self._origin_initialized:
            return None, None
        gx = int((wx - self._origin_x) / self.resolution)
        gy = int((wy - self._origin_y) / self.resolution)
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return gx, gy
        return None, None

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Return the world-frame centre of grid cell (gx, gy)."""
        wx = self._origin_x + (gx + 0.5) * self.resolution
        wy = self._origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    @property
    def origin(self) -> Tuple[float, float]:
        """Bottom-left corner of the grid in world frame."""
        return self._origin_x, self._origin_y

    @property
    def origin_initialized(self) -> bool:
        return self._origin_initialized

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _accumulate_hit(
        self, wx: float, wy: float, range_rate: float
    ) -> None:
        gx, gy = self.world_to_grid(wx, wy)
        if gx is None:
            return
        cell = self._grid[gy * self.grid_width + gx]
        cell.confidence = min(
            self.max_confidence, cell.confidence + self.hit_increment
        )
        cell.range_rate_sum += range_rate
        cell.range_rate_count += 1

    def _update_origin_locked(self, robot_x: float, robot_y: float) -> None:
        if not self._origin_initialized:
            self._origin_x = robot_x - self.map_width_m / 2.0
            self._origin_y = robot_y - self.map_height_m / 2.0
            self._origin_initialized = True
            return

        centre_x = self._origin_x + self.map_width_m / 2.0
        centre_y = self._origin_y + self.map_height_m / 2.0

        shift_x = robot_x - centre_x
        shift_y = robot_y - centre_y

        cells_x = int(round(shift_x / self.resolution))
        cells_y = int(round(shift_y / self.resolution))

        if cells_x == 0 and cells_y == 0:
            return

        self._shift_grid(cells_x, cells_y)
        self._origin_x += cells_x * self.resolution
        self._origin_y += cells_y * self.resolution

    def _shift_grid(self, dx: int, dy: int) -> None:
        """Scroll cell data by (dx, dy).  Vacated cells are zeroed."""
        new_grid = [CellData() for _ in range(self.grid_width * self.grid_height)]

        for gy in range(self.grid_height):
            src_y = gy + dy
            if src_y < 0 or src_y >= self.grid_height:
                continue
            for gx in range(self.grid_width):
                src_x = gx + dx
                if src_x < 0 or src_x >= self.grid_width:
                    continue
                new_grid[gy * self.grid_width + gx] = (
                    self._grid[src_y * self.grid_width + src_x]
                )

        self._grid = new_grid
