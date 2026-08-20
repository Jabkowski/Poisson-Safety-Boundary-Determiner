#!/usr/bin/env python

# Copyright (c) 2026
#
# This script runs a CARLA client in synchronous mode (10 Hz), spawns an ego
# vehicle with keyboard control, and renders multiple sensors including a
# top-down semantic segmentation camera (BEV) at 512x512.

import argparse
import math

import h5py
import logging
import random
import os

import carla
from occupancy_grid_map import OccupancyGridMap

try:
    import pygame
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_SPACE
    from pygame.locals import K_w
    from pygame.locals import K_r
except ImportError as exc:
    raise RuntimeError(
        "cannot import pygame, make sure pygame package is installed"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "cannot import numpy, make sure numpy package is installed"
    ) from exc

try:
    import queue
except ImportError:
    import Queue as queue  # type: ignore


class CarlaSyncMode(object):
    """Synchronize world + sensor streams under a fixed simulation time-step."""

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get("fps", 10.0)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def image_to_surface(image):
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] >= 3:
            array = array[:, :, :3]
        else:
            raise ValueError("Unsupported numpy image shape: %s" % (array.shape,))

        return pygame.surfarray.make_surface(array.swapaxes(0, 1))

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


def parse_rgb_triplet(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must have 3 comma-separated integers, e.g. 0,255,0")
    values = [int(p) for p in parts]
    for value in values:
        if value < 0 or value > 255:
            raise ValueError("Color values must be in range [0, 255]")
    return tuple(values)


class KeyboardController(object):
    """Minimal vehicle keyboard controller inspired by manual_control.py."""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.autopilot_enabled = False
        self.steer_cache = 0.0
        self.vehicle.set_autopilot(self.autopilot_enabled)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == K_ESCAPE or event.key == K_q:
                    return True
                if event.key == K_p:
                    self.autopilot_enabled = not self.autopilot_enabled
                    self.vehicle.set_autopilot(self.autopilot_enabled)
        return False

    def update_control(self, milliseconds):
        if self.autopilot_enabled:
            return

        keys = pygame.key.get_pressed()

        self.control.throttle = 0.0
        self.control.brake = 0.0
        self.control.hand_brake = False
        self.record_grid = False

        if keys[K_r]:
            self.record_grid = True
        if keys[K_w]:
            self.control.throttle = 1.0
        if keys[K_s]:
            self.control.brake = 1.0
        if keys[K_SPACE]:
            self.control.hand_brake = True

        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            self.steer_cache -= steer_increment
        elif keys[K_d]:
            self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0

        self.steer_cache = max(-1.0, min(1.0, self.steer_cache))
        self.control.steer = round(self.steer_cache, 3)
        self.vehicle.apply_control(self.control)


def select_ego_blueprint(world, actor_filter="vehicle.*"):
    candidates = world.get_blueprint_library().filter(actor_filter)
    if not candidates:
        raise RuntimeError("No vehicle blueprint matches filter: %s" % actor_filter)
    blueprint = random.choice(candidates)
    blueprint.set_attribute("role_name", "hero")
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", random.choice(colors))
    return blueprint


def spawn_ego_vehicle(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in current map.")

    random.shuffle(spawn_points)
    for transform in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            return vehicle
    raise RuntimeError("Could not spawn ego vehicle in any spawn point.")


# Radar BEV parameters
RADAR_RANGE_M = 60.0   # metres shown in half-image height
BEV_SIZE = 512         # pixels

# (location_x, location_y, yaw_deg, display_color_rgb)
_RADAR_CONFIGS = [
    ( 2.0, -1.0,  -45.0, (255,  60,  60)),   # front-left
    ( 2.0,  1.0,   45.0, ( 60,  60, 255)),   # front-right
    (-2.0, -1.0, -135.0, (255, 165,   0)),   # rear-left
    (-2.0,  1.0,  135.0, ( 60, 200,  60)),   # rear-right
]


def spawn_npc_vehicles(world, traffic_manager, count: int = 30):
    """Spawn NPC vehicles managed by the Traffic Manager."""
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    # Exclude bikes/motorbikes for cleaner radar returns
    blueprints = [b for b in blueprints if int(b.get_attribute("number_of_wheels")) >= 4]

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    npcs = []
    for transform in spawn_points[:count]:
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        npc = world.try_spawn_actor(bp, transform)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            npcs.append(npc)

    logging.info("Spawned %d NPC vehicles", len(npcs))
    return npcs


def create_sensors(world, ego_vehicle):
    blueprint_library = world.get_blueprint_library()

    # Front RGB camera.
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", "512")
    rgb_bp.set_attribute("image_size_y", "512")
    rgb_bp.set_attribute("fov", "90")
    rgb_transform = carla.Transform(
        carla.Location(x=1.5, z=2.2),
        carla.Rotation(pitch=-10),
    )
    rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)

    # 4 corner radars, 100° H-FOV each → overlapping 360° coverage.
    radar_bp = blueprint_library.find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", "100")
    radar_bp.set_attribute("vertical_fov", "10")
    radar_bp.set_attribute("range", str(RADAR_RANGE_M))
    radar_bp.set_attribute("points_per_second", "1500")

    radars = []
    for lx, ly, yaw, _ in _RADAR_CONFIGS:
        t = carla.Transform(
            carla.Location(x=lx, y=ly, z=1.0),
            carla.Rotation(yaw=yaw),
        )
        radars.append(world.spawn_actor(radar_bp, t, attach_to=ego_vehicle))

    return rgb_camera, radars


def draw_dashboard(display, font, real_fps, sim_fps, autopilot):
    lines = [
        "Real FPS: %5.1f" % real_fps,
        "Sim FPS : %5.1f" % sim_fps,
        "Mode    : %s" % ("AUTOPILOT" if autopilot else "MANUAL"),
        "Keys    : W/S throttle-brake, A/D steer, SPACE brake, P autopilot, ESC/Q quit",
    ]

    y = 8
    for line in lines:
        display.blit(font.render(line, True, (255, 255, 255)), (8, y))
        y += 18


def radar_to_bev(radar_datasets):
    """Single-frame scatter-plot BEV, colour-coded per radar."""
    bev = np.ones((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8) * 30

    scale = (BEV_SIZE / 2.0) / RADAR_RANGE_M
    cx = BEV_SIZE // 2
    cy = BEV_SIZE // 2

    for radar_data, yaw_deg, color in radar_datasets:
        yaw_rad = np.radians(yaw_deg)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)

        for detection in radar_data:
            horiz = detection.depth * np.cos(detection.altitude)
            x_local = horiz * np.cos(detection.azimuth)
            y_local = horiz * np.sin(detection.azimuth)

            fwd = x_local * cos_y - y_local * sin_y
            lat = x_local * sin_y + y_local * cos_y

            px = int(cx + lat * scale)
            py = int(cy - fwd * scale)

            if 0 <= px < BEV_SIZE and 0 <= py < BEV_SIZE:
                bev[max(0, py - 2) : py + 2, max(0, px - 2) : px + 2] = color

    bev[cy - 4 : cy + 4, cx - 4 : cx + 4] = (255, 255, 255)
    return bev


def save_radar_detections_as_h5(radar_datasets, path):
    """Append combined 4-radar detections for one frame as an HDF5 dataset.

    Each frame is stored as an Nx4 float32 array:
    columns: [azimuth_rad, altitude_rad, depth_m, velocity_m_s]
    """
    os.makedirs(path, exist_ok=True)
    h5_path = os.path.join(path, "radar.h5")

    all_points = [
        [d.azimuth, d.altitude, d.depth, d.velocity]
        for radar_data, _, _ in radar_datasets
        for d in radar_data
    ]
    points = np.array(all_points, dtype=np.float32).reshape(-1, 4)

    with h5py.File(h5_path, "a") as f:
        radar_group = f.require_group("radar")

        if len(radar_group.keys()) == 0:
            next_idx = 1
        else:
            existing_indices = sorted(int(k) for k in radar_group.keys())
            next_idx = existing_indices[-1] + 1

        key = f"{next_idx:06d}"
        radar_group.create_dataset(key, data=points)


def run(args):
    actors = []
    original_settings = None
    traffic_manager = None
    target_sync_fps = 10
    rpc_timeout = 5.0
    map_load_timeout = 60.0
    target_map = "Town03"

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((1536, 512), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(rpc_timeout)
    world = client.get_world()

    # current_map = world.get_map().name.split("/")[-1]
    # if current_map != target_map:
    #     logging.info(
    #         "Loading map %s (temporary timeout %.1fs)", target_map, map_load_timeout
    #     )
    #     client.set_timeout(map_load_timeout)
    #     world = client.load_world(target_map)
    #     world.wait_for_tick(seconds=20.0)
    #     client.set_timeout(rpc_timeout)

    try:
        original_settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(args.tm_port)

        ego_blueprint = select_ego_blueprint(world, args.filter)
        ego_vehicle = spawn_ego_vehicle(world, ego_blueprint)
        actors.append(ego_vehicle)

        controller = KeyboardController(ego_vehicle)

        npc_vehicles = spawn_npc_vehicles(world, traffic_manager, args.num_npcs)
        actors.extend(npc_vehicles)

        rgb_camera, radars = create_sensors(world, ego_vehicle)
        actors.extend([rgb_camera] + radars)

        ogm = OccupancyGridMap(
            map_width_m=60.0,
            map_height_m=60.0,
            resolution=0.4,
            decay_rate=6.0,
            hit_increment=15.0,
            max_confidence=100.0,
            range_rate_threshold=0.15,
        )

        traffic_manager.set_synchronous_mode(True)
        idx = 0
        with CarlaSyncMode(world, rgb_camera, *radars, fps=10.0) as sync_mode:
            while True:
                # Keep simulation and wall-clock aligned at 10 Hz.
                frame_ms = clock.tick_busy_loop(target_sync_fps)

                if controller.parse_events():
                    break

                controller.update_control(frame_ms)

                tick_data = sync_mode.tick(timeout=2.0)
                snapshot, rgb_image = tick_data[0], tick_data[1]
                radar_measurements = tick_data[2:]  # one per corner radar

                # Ego pose and velocity in world frame
                ego_t = ego_vehicle.get_transform()
                ego_x = ego_t.location.x
                ego_y = ego_t.location.y
                ego_yaw = math.radians(ego_t.rotation.yaw)
                cos_e, sin_e = math.cos(ego_yaw), math.sin(ego_yaw)
                ego_vel = ego_vehicle.get_velocity()

                ogm.update_origin(ego_x, ego_y)

                for meas, (lx, ly, yaw_deg, _) in zip(radar_measurements, _RADAR_CONFIGS):
                    cos_r = math.cos(math.radians(yaw_deg))
                    sin_r = math.sin(math.radians(yaw_deg))

                    for det in meas:
                        horiz = det.depth * math.cos(det.altitude)
                        xl = horiz * math.cos(det.azimuth)
                        yl = horiz * math.sin(det.azimuth)

                        # radar-local → vehicle frame (include mount offset)
                        xv = xl * cos_r - yl * sin_r + lx
                        yv = xl * sin_r + yl * cos_r + ly

                        # vehicle frame → world frame
                        wx = ego_x + xv * cos_e - yv * sin_e
                        wy = ego_y + xv * sin_e + yv * cos_e

                        rr = OccupancyGridMap.compensate_range_rate(
                            det.velocity, wx, wy,
                            ego_x, ego_y, ego_vel.x, ego_vel.y,
                        )
                        ogm.add_hit(wx, wy, rr)

                ogm.apply_decay(dt=frame_ms / 200.0)

                rgb_image.convert(carla.ColorConverter.Raw)
                bev_image = ogm.get_ego_centric_bev(
                    ego_x, ego_y, ego_yaw, RADAR_RANGE_M, (BEV_SIZE, BEV_SIZE)
                )

                radar_datasets = [
                    (meas, cfg[2], cfg[3])
                    for meas, cfg in zip(radar_measurements, _RADAR_CONFIGS)
                ]
                scatter_image = radar_to_bev(radar_datasets)

                rgb_surface     = image_to_surface(rgb_image)
                bev_surface     = image_to_surface(bev_image)
                scatter_surface = image_to_surface(scatter_image)

                display.blit(rgb_surface,     (0,    0))
                display.blit(bev_surface,     (512,  0))
                display.blit(scatter_surface, (1024, 0))

                real_fps = clock.get_fps()
                sim_fps = (
                    1.0 / snapshot.timestamp.delta_seconds
                    if snapshot.timestamp.delta_seconds > 0
                    else 0.0
                )
                draw_dashboard(
                    display, font, real_fps, sim_fps, controller.autopilot_enabled
                )
                if controller.record_grid:
                    save_radar_detections_as_h5(
                        [(meas, yaw, color) for meas, (_, _, yaw, color) in
                         zip(radar_measurements, _RADAR_CONFIGS)],
                        "/home/xk2qmc/CARLA/CARLA_0.9.16/semantic_grid_generator/grids",
                    )
                idx += 1
                pygame.display.flip()

    finally:
        if traffic_manager is not None:
            traffic_manager.set_synchronous_mode(False)
        if original_settings is not None:
            world.apply_settings(original_settings)

        for actor in actors:
            if actor is not None:
                actor.destroy()
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description="CARLA semantic BEV generator with manual ego control (sync 10 Hz)."
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "--tm-port",
        default=8000,
        type=int,
        help="Traffic Manager port (default: 8000)",
    )
    argparser.add_argument(
        "--num-npcs",
        default=30,
        type=int,
        help="Number of NPC vehicles to spawn (default: 30)",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='Ego actor filter (default: "vehicle.*")',
    )
    args = argparser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Connecting to CARLA at %s:%d", args.host, args.port)

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
