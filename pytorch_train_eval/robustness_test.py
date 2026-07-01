#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Modified version of online_model_test_CARLA from that test how resilent and robust is a neural network, 
# trained to imitate a controller.
# It implements three kinds of robustness test:
# 1. Random control test: applies random throttle and steer commands to the vehicle at regular intervals to see how the model recovers from unexpected situations3.
# 2. Position test: spawns the vehicle at different lateral positions on the road to evaluate how well the model can handle variations in the starting position.
# 3. Velocity test: spawns the vehicle with different initial velocities on the road to evaluate how well the model can handle variations in the starting velocity.

"""
Welcome to CARLA robustness test.
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# CARLA imports
import carla

from carla import ColorConverter as cc

# Generic imports
import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
import csv
import cv2 as cv
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

# Pygame imports
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Pytorch Imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
# Timm and pilotnet import
import timm
from utils.pilotnet import PilotNet

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.hud = hud
        self.player = None
        self.data_record = False
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._car_camera = None
        self.car_camera_image = [None]
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self._blueprint = args.vehicle_name
        self.spawn_points_csv = args.spawn_points_csv
        self.draw_spawn_points = args.draw_spawn_points
        self.random_control_enabled = args.random_control_test
        self.random_control_active = False
        self.random_control_start_time = 0.0
        self.random_control_interval = 20.0  # seconds between random controls
        self.random_control_duration = 0.5  # duration of random control
        self.random_control_intensity = 0.5  # intensity of the random control commmands
        # Random control test metrics tracking
        self.random_control_initial_position = None  # Initial spawn position
        self.random_control_start_delay = 2.0  # Wait 2 seconds after spawn before counting laps
        self.random_control_lap_threshold = 5.0  # Distance threshold to consider returned to start
        self.laps_completed = 0  # Number of completed laps
        self.has_left_start_area = False  # Track if car has left initial position area
        self.test_start_time = 0.0  # Test start time
        self.test_duration = 0.0  # Duration of test until collision
        self.spawn_point_idx = 0  # sequential spawn point index
        self.position_test_enabled = args.position_test  # position test flag
        self.position_offset_idx = 0  # index for lateral position offsets [-5, 0, 5, 10]
        self.position_offsets = [-3.0,-2.0, 0.0]  # lateral offsets in meters
        self.yaw_offsets = [-20.0,-10.0, 10.0, 20.0]
        # Position test metrics tracking
        self.position_test_initial_yaw = 0.0 # Initial yaw when test starts
        self.position_test_initial_lateral_offset = 0.0  # Initial lateral offset
        self.position_test_start_time = 0.0  # Simulation time when test restarts
        self.recovery_time = None  # Time to recover orientation
        self.orientation_threshold = 0.5  # Degrees within which we consider orientation recovered
        self.lateral_offset_threshold = 0.5  # Meters within which we consider lateral position recovered
        self.position_test_metrics = []  # List of metrics for each restart

        # Velocity test metrics tracking
        self.velocity_test_enabled = args.velocity_test
        self.velocity_test_initial_velocity = 0.0  # Initial velocity applied
        self.velocity_recovery_time = None  # Time to recover to default velocity
        self.velocity_recovery_threshold = 1.0  # km/h threshold to consider recovered
        self.default_target_velocity = 25.0  # Default target velocity in km/h
        self.velocity_test_metrics = []  # List of metrics for each restart

        # Random control test metrics tracking
        self.random_control_test_metrics = []  # List of metrics for each restart
        self.max_laps = 5  # Number of laps to complete for each random control test
        self.current_lap_start_time = 0.0  # Simulation time when current lap started
        self.lap_positions = []  # List of (x, y) positions recorded during current lap
        self.waypoints = []  # Waypoints loaded from CSV file
        self.waypoints_file = args.waypoints_csv if hasattr(args, 'waypoints_csv') else None  # Path to waypoints CSV
        self.lane_invasion_count = 0  # Total lane invasions during random control test
        self.current_lap_lane_invasions = 0  # Lane invasions in current lap
        self.deviation_executor = ThreadPoolExecutor(max_workers=1)  # Limit to 1 thread for deviation calculations to prevent resource exhaustion

        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

        if self.random_control_enabled:
            self.max_restarts = 1
            # Load waypoints if file exists
            if self.waypoints_file:
                self.waypoints = self._load_waypoints(self.waypoints_file)
        elif self.position_test_enabled:
            self.max_restarts = len(self.position_offsets) * len(self.yaw_offsets) - 1 # -1 because the first restart is the default position
        elif self.velocity_test_enabled:
            self.max_restarts = len(load_spawn_points(self.spawn_points_csv)) - 1 # -1 because the first restart is the default position


    def _load_waypoints(self, file_path):
        """Load waypoints from CSV file.
        
        Args:
            file_path: Path to CSV file with columns: index, x, y
            
        Returns:
            List of tuples (x, y) for each waypoint
        """
        waypoints = []
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 3:
                        waypoints.append((float(row[1]), float(row[2])))
            print(f"Loaded {len(waypoints)} waypoints from {file_path}")
        except Exception as e:
            print(f"Error loading waypoints: {e}")
        return waypoints

    def calculate_path_deviation(self, recorded_positions):
        """Calculate deviation between recorded path and waypoints using vectorized numpy operations.
        
        Args:
            recorded_positions: List of (x, y) positions recorded during the lap
            
        Returns:
            Dictionary with deviation metrics (mean, max, std_dev)
        """
        if not self.waypoints or not recorded_positions:
            return {'mean': 0.0, 'max': 0.0, 'std_dev': 0.0}
        
        # Convert to numpy arrays for vectorized operations (much faster than nested loops)
        rec_pos_array = np.array(recorded_positions, dtype=np.float32)  # Shape: (n, 2)
        way_pos_array = np.array(self.waypoints, dtype=np.float32)  # Shape: (m, 2)
        
        # Compute pairwise distances using broadcasting: (n, m, 2) -> (n, m)
        # This is much faster than nested Python loops
        diff = rec_pos_array[:, np.newaxis, :] - way_pos_array[np.newaxis, :, :]  # Shape: (n, m, 2)
        distances = np.sqrt((diff ** 2).sum(axis=2))  # Shape: (n, m)
        
        # Find minimum distance (closest waypoint) for each recorded position
        deviations = distances.min(axis=1)  # Shape: (n,)
        
        if len(deviations) > 0:
            mean_dev = float(np.mean(deviations))
            max_dev = float(np.max(deviations))
            std_dev = float(np.std(deviations))
            return {'mean': mean_dev, 'max': max_dev, 'std_dev': std_dev}
        
        return {'mean': 0.0, 'max': 0.0, 'std_dev': 0.0}

    def calculate_path_deviation_threaded(self, recorded_positions, metric_entry):
        """Calculate path deviation in a separate thread to avoid blocking model inference.
        
        Args:
            recorded_positions: List of (x, y) positions recorded during the lap
            metric_entry: Dictionary to store the results (passed by reference)
        """
        deviation_metrics = self.calculate_path_deviation(recorded_positions)
        metric_entry['path_deviation_mean'] = deviation_metrics['mean']
        metric_entry['path_deviation_max'] = deviation_metrics['max']
        metric_entry['path_deviation_stddev'] = deviation_metrics['std_dev']
        metric_entry['deviation_ready'] = True
        print(f"  Path deviation calculated - Mean: {deviation_metrics['mean']:.2f}m, Max: {deviation_metrics['max']:.2f}m, StdDev: {deviation_metrics['std_dev']:.2f}m")

    def camera_callback(self, image, return_image):
        return_image[0] = image

    def restart(self):
        #Set default weather
        self.world.set_weather(carla.WeatherParameters.CloudyNoon)
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_library = self.world.get_blueprint_library()
        if self._blueprint is None:
            blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)     
            if not blueprint_list:
                raise ValueError("Couldn't find any blueprints with the specified filters")
            blueprint = random.choice(blueprint_list)
        else:
            blueprint = blueprint_library.find(self._blueprint)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            # Create route from the chosen spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            route_1_indices = load_spawn_points(self.spawn_points_csv)

            # Draw spawn points
            if self.draw_spawn_points:
                for ind in route_1_indices:
                    self.world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60000, color=carla.Color(255,0,0))

            # We choose the next spawn point sequentially from the route.
            init_spawn_point =  spawn_points[route_1_indices[self.spawn_point_idx]]

            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, init_spawn_point)
        while self.player is None:
            # Create route from the chosen spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            route_1_indices = load_spawn_points(self.spawn_points_csv)
            # We choose the next spawn point sequentially from the route.
            init_spawn_point =  spawn_points[route_1_indices[self.spawn_point_idx]]
            self.player = self.world.try_spawn_actor(blueprint, init_spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud, self)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.car_camera_image = [None]
        # camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self._car_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
        print('created %s' % self._car_camera.type_id)
        self._car_camera.listen(lambda image: self.camera_callback(image, self.car_camera_image))
        
        # Apply lateral position offset if position_test is enabled
        if self.position_test_enabled:

            current_transform = self.player.get_transform()
            yaw = math.radians(current_transform.rotation.yaw)
            # Lateral direction is perpendicular to forward direction
            lateral_direction = carla.Vector3D(-math.sin(yaw), math.cos(yaw), 0.0)
            # Get the offset for this restart
            offset_value = self.position_offsets[self.position_offset_idx // len(self.yaw_offsets)]
            # Calculate the new location
            new_location = current_transform.location + lateral_direction * offset_value

            # Apply random yaw rotation between -25 and 25 degrees
            new_yaw = current_transform.rotation.yaw + self.yaw_offsets[self.position_offset_idx % len(self.yaw_offsets)]
            new_rotation = carla.Rotation(pitch=current_transform.rotation.pitch, 
                                         yaw=new_yaw, 
                                         roll=current_transform.rotation.roll)
            new_transform = carla.Transform(new_location, new_rotation)
            self.player.set_transform(new_transform)
            print(f"Position test: Applied lateral offset of {offset_value}m and rotation of {new_yaw:.1f}°")
            # Store initial conditions for metrics tracking
            self.position_test_initial_yaw = new_yaw
            self.position_test_initial_lateral_offset = offset_value
            self.recovery_time = None
            # Increment offset index for next restart
            self.position_offset_idx = self.position_offset_idx + 1
        else:
            # Increment spawn point index for next restart, wrapping around
            spawn_points = self.world.get_map().get_spawn_points()
            route_1_indices = load_spawn_points(self.spawn_points_csv)
            self.spawn_point_idx = (self.spawn_point_idx + 1) % len(route_1_indices)
        
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def apply_random_velocity(self):
        """Apply a random target velocity to the vehicle in the direction it is facing."""
        random_speed = random.uniform(10.0, 30.0)  # Speed in km/h
        # Get the direction the vehicle is facing from its rotation
        transform = self.player.get_transform()
        yaw = math.radians(transform.rotation.yaw)  # Convert degrees to radians
        forward_direction = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
        self.player.set_target_velocity(forward_direction * random_speed)

    def get_orientation_and_lateral_position_offset(self):
        """Get the vehicle's current yaw angle and lateral position offset.
        
        Returns:
            tuple: (yaw_angle in degrees, lateral_offset in meters)
        """
        transform = self.player.get_transform()
        yaw = transform.rotation.yaw

        # To get lateral position, we need to calculate distance from the road center
        # Get vehicle location
        location = transform.location
        
        # Get the spawn point to establish reference
        spawn_points = self.world.get_map().get_spawn_points()
        route_1_indices = load_spawn_points(self.spawn_points_csv)
        spawn_point_location = spawn_points[route_1_indices[self.spawn_point_idx]].location

        # Calculate lateral direction perpendicular to the spawn point's forward direction
        spawn_yaw_rad = math.radians(spawn_points[route_1_indices[self.spawn_point_idx]].rotation.yaw)
        lateral_direction = carla.Vector3D(-math.sin(spawn_yaw_rad), math.cos(spawn_yaw_rad), 0.0)
        
        # Calculate offset from spawn point
        delta = location - spawn_point_location
        lateral_offset = delta.x * lateral_direction.x + delta.y * lateral_direction.y
        
        # Calculate yaw offset from the correct orientation
        yaw_offset = spawn_points[route_1_indices[self.spawn_point_idx]].rotation.yaw - yaw

        return yaw_offset, lateral_offset

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self._car_camera]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        # Shutdown the thread pool executor
        self.deviation_executor.shutdown(wait=True)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud, world=None):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        self.world = world  # Reference to world for tracking invasions during random control test
        carla_world = self._parent.get_world()
        bp = carla_world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = carla_world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))
        # Track lane invasions during random control test
        if self.world and self.world.random_control_enabled:
            self.world.lane_invasion_count += 1
            self.world.current_lap_lane_invasions += 1

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.marestartke_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

def load_spawn_points(file_path):
  """Loads data from a CSV file into a list of lists.

  Args:
    file_path: The path to the CSV file.

  Returns:
    A list of lists, where each inner list represents a row in the CSV file.
  """
  data = []
  with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row:
            data.append(int(row[0]))
  return data


# ==============================================================================
# -- carla_cam_to_image() ---------------------------------------------------------------
# ==============================================================================


def carla_cam_to_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3] 


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):

    pygame.init()
    pygame.font.init()
    world = None

    
    image_shape = (66, 200, 3)
    input_size =[66, 200]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
    
    # Load the state dictionary from the local .pth file
    state_dict = torch.load(args.model_path,weights_only=True)
    model_name = args.model

    if model_name == 'pilotnet':
        model = PilotNet(image_shape, 2)
    elif model_name == 'mobilenet_large':
        model = models.mobilenet_v3_large()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'mobilenet_small':
        model = models.mobilenet_v3_small()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet_v2':
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientvit':
        model = timm.create_model('efficientvit_b0', pretrained=False)
        num_ftrs = model.head.classifier[-1].in_features
        model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'fastvit':
        model = timm.create_model('fastvit_sa12', pretrained=False)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, 2)
    else:
        print("Model not found")
        exit()
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the selected device (cpu or gpu)
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    print("Model loaded successfully")


    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        client.load_world_if_different(args.town_name)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        
        clock = pygame.time.Clock()

        max_restarts = world.max_restarts # number of restarts before ending the test, set to -1 for infinite restarts
        random_control_timer = 0.0 # timer for random control intervals
        collision_count = 0  # Track number of collisions to detect new ones
        test_collisions = 0 # Number of total collisions during the test
        test_timer = 0.0  # Timer for position and velocity tests
        test_interval = args.test_time  # Restart every 20 seconds during tests
        restart_count = 0  # Track number of restarts for position/velocity tests
        test_restart_time = world.hud.simulation_time  # Track simulation time of current restart

        while True:
            clock.tick_busy_loop(60)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            image = world.car_camera_image[0]
            
            # Timer for random control and tests
            delta_time = clock.get_time() / 1000.0  # convert to seconds
            random_control_timer += delta_time
            test_timer += delta_time
            
            is_random_control_active = False
            #Random control logic
            if world.random_control_enabled:
                if random_control_timer >= world.random_control_interval:
                    world.random_control_active = True
                    world.random_control_start_time = random_control_timer
                    random_control_timer = 0.0
                if world.random_control_active:
                    elapsed = random_control_timer - (world.random_control_start_time - world.random_control_interval)
                    if elapsed < world.random_control_duration:
                        is_random_control_active = True
                        # Apply random control
                        random_control = carla.VehicleControl()
                        random_control.throttle = random.uniform(0.5, 0.5 + world.random_control_intensity)
                        random_control.steer = random.uniform(-world.random_control_intensity, world.random_control_intensity)
                        world.player.apply_control(random_control)
                else:
                    world.random_control_active = False
            
            # Position test metrics
            if args.position_test and world.position_test_enabled:
                time_since_restart = world.hud.simulation_time - test_restart_time
                current_yaw_offset, current_lateral_offset = world.get_orientation_and_lateral_position_offset()

                if abs(current_yaw_offset) < world.orientation_threshold and abs(current_lateral_offset) < world.lateral_offset_threshold:
                    if world.recovery_time is None and time_since_restart > 0.5:  # Add a small buffer to avoid false positives immediately after restart
                        world.recovery_time = time_since_restart
                        print(f"Recovery achieved in {world.recovery_time:.2f}s")

            # Velocity test metrics
            if args.velocity_test and world.velocity_test_enabled:
                time_since_restart = world.hud.simulation_time - test_restart_time
                v = world.player.get_velocity()
                current_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

                if abs(current_velocity - world.default_target_velocity) < world.velocity_recovery_threshold:
                    if world.velocity_recovery_time is None and time_since_restart > 0.5:  # Add buffer to avoid false positives
                        world.velocity_recovery_time = time_since_restart
                        print(f"Velocity recovery achieved in {world.velocity_recovery_time:.2f}s (current: {current_velocity:.2f} km/h)")

            # Random control test metrics - lap tracking
            if args.random_control_test and world.random_control_enabled:
                # Initialize on first iteration
                if world.random_control_initial_position is None:
                    world.random_control_initial_position = world.player.get_location()
                    world.test_start_time = world.hud.simulation_time

                # Wait before counting laps to avoid false positives at spawn
                time_since_test_start = world.hud.simulation_time - world.test_start_time

                if time_since_test_start > world.random_control_start_delay:
                    current_location = world.player.get_location()
                    distance_from_start = current_location.distance(world.random_control_initial_position)

                    # Check if car has left start area
                    if distance_from_start > world.random_control_lap_threshold:
                        world.has_left_start_area = True
                        # Initialize lap start time on first departure
                        if world.current_lap_start_time == 0.0:
                            world.current_lap_start_time = world.hud.simulation_time
                        # Record position during lap
                        if world.laps_completed < world.max_laps:  # Only record while completing laps
                            world.lap_positions.append((current_location.x, current_location.y))

                    # Count lap when returning to start after leaving
                    if world.has_left_start_area and distance_from_start < world.random_control_lap_threshold:
                        world.laps_completed += 1
                        world.has_left_start_area = False
                        # Calculate lap time
                        lap_time = world.hud.simulation_time - world.current_lap_start_time
                        print(f"Lap {world.laps_completed} completed in {lap_time:.2f}s!")
                        
                        # Create metric entry with placeholder values (will be filled by thread)
                        metric_entry = {
                            'lap_num': world.laps_completed,
                            'lap_time': lap_time,
                            'lane_invasions': world.current_lap_lane_invasions,
                            'path_deviation_mean': 0.0,
                            'path_deviation_max': 0.0,
                            'path_deviation_stddev': 0.0,
                            'deviation_ready': False
                        }
                        world.random_control_test_metrics.append(metric_entry)
                        
                        # Start deviation calculation in thread pool (queued, limited concurrency to prevent slowdown)
                        positions_copy = list(world.lap_positions)
                        world.deviation_executor.submit(
                            world.calculate_path_deviation_threaded,
                            positions_copy,
                            metric_entry
                        )
                        
                        # Reset lap start time, positions, and lane invasions for next lap
                        world.current_lap_start_time = world.hud.simulation_time
                        world.lap_positions = []
                        world.current_lap_lane_invasions = 0
                        # End test if 5 laps completed
                        if world.laps_completed >= world.max_laps:
                            world.test_duration = world.hud.simulation_time - world.test_start_time
                            print(f"\n=== RANDOM CONTROL TEST SUMMARY ===")
                            print(f"Test duration: {world.test_duration:.2f}s")
                            print(f"Laps completed: {world.laps_completed}")
                            
                            # Wait for all deviation calculations to complete
                            print("Waiting for deviation calculations to complete...")
                            while True:
                                all_ready = True
                                for metric in world.random_control_test_metrics:
                                    if not metric.get('deviation_ready', False):
                                        all_ready = False
                                        break
                                if all_ready:
                                    break
                                time.sleep(0.1)
                            
                            # Print lap times and deviations
                            avg_lap_time = 0
                            avg_lane_invasions = 0
                            for metric in world.random_control_test_metrics:
                                avg_lap_time += metric['lap_time']
                                avg_lane_invasions += metric['lane_invasions']
                            if world.random_control_test_metrics:
                                avg_lap_time /= len(world.random_control_test_metrics)
                                avg_lane_invasions /= len(world.random_control_test_metrics)
                                avg_deviation = 0
                                for metric in world.random_control_test_metrics:
                                    avg_deviation += metric['path_deviation_mean']
                                avg_deviation /= len(world.random_control_test_metrics)
                                
                                print(f"Average lap time: {avg_lap_time:.2f}s")
                                print(f"Average lane invasions per lap: {avg_lane_invasions:.2f}")
                                print(f"Total lane invasions: {world.lane_invasion_count}")
                                print(f"Average path deviation: {avg_deviation:.2f}m")
                                for metric in world.random_control_test_metrics:
                                    print(f"  Lap {metric['lap_num']}: {metric['lap_time']:.2f}s, Lane invasions: {metric['lane_invasions']}, Deviation: {metric['path_deviation_mean']:.2f}m")
                            break
            if image is not None and not is_random_control_active:
                image = carla_cam_to_image(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                cropped_image = image[240:480, 0:640]
                resized_image = cv.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

                input_tensor = preprocess(resized_image).to(device)
                input_batch = input_tensor.unsqueeze(0)

                output = model(input_batch)
                if device == "cpu":
                    net_throttle = output[0].detach().numpy()[0].item()
                    net_steer = output[0].detach().numpy()[1].item()
                    #net_brake = output[0].detach().numpy()[2].item()
                else:
                    net_throttle = output.data.cpu().numpy()[0][0].item()
                    net_steer = output.data.cpu().numpy()[0][1].item()
                    #net_brake = output.data.cpu().numpy()[0][2].item()
                
                control = carla.VehicleControl()
                control.throttle = net_throttle
                control.steer = net_steer
                #control.brake = 0.0
                control.manual_gear_shift=True
                if control.throttle < 0:
                    control.gear = -1
                    control.throttle = -control.throttle
                else:
                    control.gear = 1
                #print(control)
                world.player.apply_control(control)
            
            # Check for collision and restart if one detected
            current_collisions = len(world.collision_sensor.history)
            should_restart = False
            restart_reason = ""

            if current_collisions > collision_count:
                collision_count = current_collisions
                if args.random_control_test:
                    # End random control test on collision
                    world.test_duration = world.hud.simulation_time - world.test_start_time
                    print(f"\n=== RANDOM CONTROL TEST SUMMARY ===")
                    print(f"Test duration: {world.test_duration:.2f}s")
                    print(f"Total collisions: {collision_count}")
                    print(f"Laps completed: {world.laps_completed}")
                    
                    # Wait for all deviation calculations to complete
                    if world.random_control_test_metrics:
                        print("Waiting for deviation calculations to complete...")
                        while True:
                            all_ready = True
                            for metric in world.random_control_test_metrics:
                                if not metric.get('deviation_ready', False):
                                    all_ready = False
                                    break
                            if all_ready:
                                break
                            time.sleep(0.1)
                    break
                else:
                    should_restart = True
                    restart_reason = f"Collision detected! Total collisions: {collision_count}"
                    test_collisions += 1
            
            # Check for time-based restart during position or velocity tests
            if (args.position_test or args.velocity_test) and test_timer >= test_interval:
                should_restart = True
                restart_reason = f"Time limit reached"
                test_timer = 0.0

            if should_restart:
                print(restart_reason)
                
                # Record position test metrics
                if args.position_test:
                    test_restart_time = world.hud.simulation_time  # Reset position test timer on restart
                    metric_entry = {
                        'restart_num': restart_count,
                        'initial_yaw': world.position_test_initial_yaw,
                        'initial_lateral_offset': world.position_test_initial_lateral_offset,
                        'recovery_time': world.recovery_time,
                    }
                    world.position_test_metrics.append(metric_entry)
                    print(f"Position test restart #{restart_count}:")
                    print(f"  Initial yaw: {world.position_test_initial_yaw:.2f}°, Initial lateral offset: {world.position_test_initial_lateral_offset:.2f}m")
                    if world.recovery_time is not None:
                        print(f"  Recovery Time: {world.recovery_time:.2f}s")
                    else:
                        print(f"  Recovery not achieved")

                # Record velocity test metrics
                if args.velocity_test:
                    test_restart_time = world.hud.simulation_time
                    metric_entry = {
                        'restart_num': restart_count,
                        'initial_velocity': world.velocity_test_initial_velocity,
                        'recovery_time': world.velocity_recovery_time,
                    }
                    world.velocity_test_metrics.append(metric_entry)
                    print(f"Velocity test restart #{restart_count}:")
                    print(f"  Initial velocity: {world.velocity_test_initial_velocity:.2f} km/h")
                    if world.velocity_recovery_time is not None:
                        print(f"  Recovery Time: {world.velocity_recovery_time:.2f}s")
                    else:
                        print(f"  Recovery not achieved")

                    world.apply_random_velocity()
                    v = world.player.get_velocity()
                    world.velocity_test_initial_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                    world.velocity_recovery_time = None
                    print(f"Applied velocity to vehicle: {world.velocity_test_initial_velocity:.2f} km/h")
    
                if (args.position_test or args.velocity_test) and restart_count >= max_restarts:
                    print(f"Max restarts reached. Ending test.")
                    print(f"Total collisions of the test: {test_collisions}")

                    # Print final summary of position test metrics
                    if args.position_test:
                        print("\n=== POSITION TEST SUMMARY ===")
                        print(f"Total restarts: {restart_count}")
                        avg_recovery_time = 0
                        recovery_success = 0
                        for metric in world.position_test_metrics:
                            if metric['recovery_time'] is not None:
                                avg_recovery_time += metric['recovery_time']
                                recovery_success += 1

                        if recovery_success > 0:
                            avg_recovery_time /= recovery_success
                            print(f"Average recovery time: {avg_recovery_time:.2f}s ({recovery_success}/{restart_count+1} successful)")
                        else:
                            print(f"Recovery: No successful recoveries")

                    # Print final summary of velocity test metrics
                    if args.velocity_test:
                        print("\n=== VELOCITY TEST SUMMARY ===")
                        print(f"Total restarts: {restart_count}")
                        avg_recovery_time = 0
                        recovery_success = 0
                        for metric in world.velocity_test_metrics:
                            if metric['recovery_time'] is not None:
                                avg_recovery_time += metric['recovery_time']
                                recovery_success += 1

                        if recovery_success > 0:
                            avg_recovery_time /= recovery_success
                            print(f"Average recovery time: {avg_recovery_time:.2f}s ({recovery_success}/{restart_count+1} successful)")
                        else:
                            print(f"Recovery: No successful recoveries")
                    break
                world.restart()
                restart_count += 1
            
    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# Function to close CARLA simulation
def close_carla():

    try:
        ps_output = subprocess.check_output(["ps", "-Af"]).decode('utf-8').strip("\n")
    except subprocess.CalledProcessError as ce:
        logging.error("SimulatorEnv: exception raised executing ps command {}".format(ce))
        sys.exit(-1)


    if ps_output.count('CarlaUE4.sh') > 0:
        # kill zombies processes -> nohup ./CarlaUE4.sh > /dev/null 2>&1 &
        try:
            subprocess.check_call(["killall", "-9", "CarlaUE4.sh"])
            logging.debug("SimulatorEnv: CARLA server killed.")
        except subprocess.CalledProcessError as ce:
            logging.error("SimulatorEnv: exception raised executing killall command for CARLA server {}".format(ce))

    if ps_output.count('CarlaUE4-Linux-Shipping') > 0:
        try:
            subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-Shipping"])
            logging.debug("SimulatorEnv: CarlaUE4-Linux-Shipping killed.")
        except subprocess.CalledProcessError as ce:
            logging.error("SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux-Shipping {}".format(ce))



# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument("--spawn_points_csv", type=str,default="./Town01_spawn_points.csv", help="File with the spawn points of the CARLA map")
    argparser.add_argument("--draw_spawn_points", type=bool,default=False, help="Enable or disable the visibility of the spawn points")
    argparser.add_argument("--vehicle_name", type=str,default="vehicle.mercedes.coupe_2020", help="Car model to load")   
    argparser.add_argument("--town_name", type=str,default="Town01", help="Carla Map to load")
    argparser.add_argument("--model", type=str, default='pilotnet', help="Model type")
    argparser.add_argument("--model_path", type=str, help="Path to the saved model")
    argparser.add_argument("--test_time", type=float, default=15.0, help="Time to restarrt the simulation")
    argparser.add_argument("--position_test", type=bool, default=False, help="Enable different starting position to test robustness")
    argparser.add_argument("--velocity_test", type=bool, default=False, help="Enable different starting velocity to test robustness")
    argparser.add_argument("--random_control_test", type=bool, default=False, help="Enable random control to test robustness")
    argparser.add_argument("--waypoints_csv", type=str, default="./carla_map_01_waypoints.csv", help="Waypoints file")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)


    carla_bin = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
    carla_root = os.environ.get("CARLA_ROOT")
        
  
    with open("/tmp/.carla_stdout.log", "w") as out, open("/tmp/.carla_stderr.log", "w") as err:
        subprocess.Popen([carla_bin, "-RenderOffScreen", "-prefernvidia"],  # "/bin/bash", 
                         cwd=carla_root,
                         stdout=out, stderr=err,
                         shell=False,
                         env=os.environ
                         )
    logging.info("CARLA server started")
    time.sleep(10)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        logging.info("Finishing CAARLA server")
        close_carla()




if __name__ == '__main__':

    main()
