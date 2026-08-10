#!/usr/bin/env python

# Copyright (c) 2026
#
# This script runs a CARLA client in synchronous mode (10 Hz), spawns an ego
# vehicle with keyboard control, and renders multiple sensors including a
# top-down semantic segmentation camera (BEV) at 512x512.

import argparse

import h5py
from utils import get_cityscapes_color
from utils import CityObjectLabel
import logging
import random
import os

import carla

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


def create_sensors(world, ego_vehicle):
    blueprint_library = world.get_blueprint_library()

    # Front RGB camera (second sensor to satisfy multi-sensor requirement).
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", "512")
    rgb_bp.set_attribute("image_size_y", "512")
    rgb_bp.set_attribute("fov", "90")

    rgb_transform = carla.Transform(
        carla.Location(x=1.5, z=2.2),
        carla.Rotation(pitch=-10),
    )
    rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)

    # Top-down semantic segmentation camera (BEV), square 512x512.
    sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", "512")
    sem_bp.set_attribute("image_size_y", "512")
    sem_bp.set_attribute("fov", "140")

    sem_bev_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=7.0),
        carla.Rotation(pitch=-90.0),
    )
    sem_camera = world.spawn_actor(sem_bp, sem_bev_transform, attach_to=ego_vehicle)

    return rgb_camera, sem_camera


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


def convert_to_occupancy_like(
    image,
):
    """Convert carla.ColorConverter.CityScapesPalette image to a simplified occupancy-like format image where white is free space and black is occupied space."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]

    # Define a mapping from CityObjectLabel to occupancy-like values.
    occupancy_map = {
        CityObjectLabel.Buildings: 1,
        CityObjectLabel.Fences: 2,
        # CityObjectLabel.Other: 3,
        CityObjectLabel.Pedestrians: 4,
        CityObjectLabel.Poles: 5,
        # CityObjectLabel.RoadLines: 6,
        # CityObjectLabel.Roads: 7,
        # CityObjectLabel.Sidewalks: 8,
        CityObjectLabel.Vegetation: 9,
        CityObjectLabel.Vehicles: 10,
        CityObjectLabel.Walls: 11,
        CityObjectLabel.TrafficSigns: 12,
        CityObjectLabel.Sky: 13,
        # CityObjectLabel.Ground: 14,
        CityObjectLabel.Bridge: 15,
        CityObjectLabel.RailTrack: 16,
        CityObjectLabel.GuardRail: 17,
        # CityObjectLabel.TrafficLight: 18,
        CityObjectLabel.Static: 19,
        # CityObjectLabel.Dynamic: 20,
        CityObjectLabel.Water: 21,
        CityObjectLabel.Terrain: 22,
    }
    # change color pixels which are occupancy into one same color (e.g. black) and non-occupancy color pixels into another color (e.g. white)
    occupancy_image = np.ones((image.height, image.width), dtype=np.uint8) * 255
    for label in occupancy_map:
        occupancy_color = get_cityscapes_color(label)
        occupancy_image[np.all(rgb == occupancy_color, axis=-1)] = 0
    return occupancy_image


def save_occupancy_image_as_h5(occupancy_image, path):
    """
    Append occupancy image as a new grid into an HDF5 file.
        Expected occupancy 0 free 1 on, black is occupied need to be saved as 0, white is free needs to be saved as 1.
    - Creates file if not exists
    - Reuses existing 'grid' group
    - Automatically assigns next key: '000001', '000002', ...
    """

    os.makedirs(path, exist_ok=True)
    h5_path = os.path.join(path, "grid.h5")

    # Convert to 0/1 uint8
    grid = np.array(
        [[1 if pixel == 0 else 0 for pixel in row] for row in occupancy_image],
        dtype=np.uint8,
    )

    with h5py.File(h5_path, "a") as f:
        # Ensure group exists
        grid_group = f.require_group("grid")

        # Determine next index
        if len(grid_group.keys()) == 0:
            next_idx = 1
        else:
            existing_indices = sorted(int(k) for k in grid_group.keys())
            next_idx = existing_indices[-1] + 1

        key = f"{next_idx:06d}"  # e.g., '000001'

        grid_group.create_dataset(key, data=grid)


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
    display = pygame.display.set_mode((1024, 512), pygame.HWSURFACE | pygame.DOUBLEBUF)
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

        rgb_camera, sem_camera = create_sensors(world, ego_vehicle)
        actors.extend([rgb_camera, sem_camera])

        traffic_manager.set_synchronous_mode(True)
        idx = 0
        with CarlaSyncMode(world, rgb_camera, sem_camera, fps=10.0) as sync_mode:
            while True:
                # Keep simulation and wall-clock aligned at 10 Hz.
                frame_ms = clock.tick_busy_loop(target_sync_fps)

                if controller.parse_events():
                    break

                controller.update_control(frame_ms)

                snapshot, rgb_image, sem_image = sync_mode.tick(timeout=2.0)

                sem_image.convert(carla.ColorConverter.CityScapesPalette)
                rgb_image.convert(carla.ColorConverter.Raw)
                sem_image = convert_to_occupancy_like(sem_image)
                rgb_surface = image_to_surface(rgb_image)
                sem_surface = image_to_surface(sem_image)

                display.blit(rgb_surface, (0, 0))
                display.blit(sem_surface, (512, 0))

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
                    save_occupancy_image_as_h5(
                        sem_image,
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
