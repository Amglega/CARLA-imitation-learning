#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import numpy as np
import random
import carla
import time
import cv2 as cv
import csv
import argparse
import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from utils.pilotnet import PilotNet

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def set_weather_atributes(self, clouds=0, rain=0, rain_deposits=0, wind=0, fog=0, wet=0, sun_azimuth=0, sun_altitude=0):
        
        self.weather.cloudiness = clouds
        self.weather.precipitation = rain
        self.weather.precipitation_deposits = rain_deposits
        self.weather.wind_intensity = wind
        self.weather.fog_density = fog
        self.weather.wetness = wet
        self.weather.sun_azimuth_angle = sun_azimuth
        self.weather.sun_altitude_angle = sun_altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def create_dataset_directory(path,dir_name="CARLA_dataset_dagger"):
  i = 0
  while True:
    directory_name = f"{path}/{dir_name}_{i}"
    if not os.path.exists(directory_name):
      os.makedirs(directory_name)
      print(f"Created directory: {directory_name}")
      return directory_name
    i += 1

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


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))


def camera_callback(image, return_image):
    return_image[0] = image

def carla_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, :3]  


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--town_name", type=str,default="Town01", help="Carla Map to load")
    parser.add_argument("--spawn_points_csv", type=str,default="./Town01_spawn_points.csv", help="File qith the spawn points of the CARLA map")
    parser.add_argument("--draw_spawn_points", type=bool,default=False, help="Enable or disable the visibility of the spawn points")
    parser.add_argument("--vehicle_name", type=str,default='vehicle.mercedes.coupe_2020', help="Car model to load")
    
    args = parser.parse_args()
    return args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='pilotnet', help="Model name")
    parser.add_argument("--saved_model_path", type=str, help="Path to the saved model")
    parser.add_argument("--town_name", type=str,default="Town01", help="Carla Map to load")
    parser.add_argument("--spawn_points_csv", type=str,default="./Town01_spawn_points.csv", help="File qith the spawn points of the CARLA map")
    parser.add_argument("--draw_spawn_points", type=bool,default=False, help="Enable or disable the visibility of the spawn points")
    parser.add_argument("--vehicle_name", type=str,default='vehicle.mercedes.coupe_2020', help="Car model to load")
    
    args = parser.parse_args()
    return args


def main():
    

    args = parse_args()
    
    image_shape = (66, 200, 3)
    num_labels = 2
    input_size =[66, 200]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
    
    # Load the state dictionary from the local .pth file
    state_dict = torch.load(args.saved_model_path)
    model_name = args.model_name

    if model_name == 'pilotnet':
        model = PilotNet(image_shape, 3).to(device)
    elif model_name == 'mobilenet_large':
        model = models.mobilenet_v3_large().to(device)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'mobilenet_small':
        model = models.mobilenet_v3_small().to(device)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'resnet':
        model = models.resnet18().to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    print("Model loaded successfully")
    
    args = parse_args()

    actor_list = []
    iteration = 0
    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(200.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        client.load_world_if_different(args.town_name)
        world = client.get_world()

        weather = Weather(world.get_weather())

        weather.set_weather_atributes(clouds=20, sun_azimuth=210, sun_altitude=20)
        world.set_weather(weather.weather)
               

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = blueprint_library.find(args.vehicle_name)
        
        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Create route from the chosen spawn points
        spawn_points = world.get_map().get_spawn_points()
        route_1_indices = load_spawn_points(args.spawn_points_csv)

        # Draw spawn points
        if args.draw_spawn_points:
            for ind in route_1_indices:
                spawn_points[ind].location
                world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60000, color=carla.Color(255,0,0))

        # We choose a random spawn point from the route.
        #spawn_point_idx = random.randint(0, len(route_1_indices) - 1)
        #spawn_point_1 =  spawn_points[route_1_indices[spawn_point_idx]]

        spawn_point_idx = 0
        spawn_point_1 =  spawn_points[route_1_indices[spawn_point_idx]]

        # We tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, spawn_point_1)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's add now a "rgb" camera attached to the vehicle.
        camera_img = [None]
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        camera.listen(lambda image: camera_callback(image, camera_img))

        spectator = world.get_spectator()
        spectator.set_transform(get_transform(vehicle.get_location(), -90))

    
        frequency = 60

        while 1 and iteration < 1000:
        
            init_time = time.time()
            image = camera_img[0]
            if image is not None:
                image = carla_to_rgb(image)
                cropped_image = image[240:480, 0:640]

                resized_image = cv.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

                input_tensor = preprocess(resized_image).to(device)
                input_batch = input_tensor.unsqueeze(0)

                output = model(input_batch)
                if device == "cpu":
                    net_throttle = output[0].detach().numpy()[0].item()
                    net_steer = output[0].detach().numpy()[1].item()
                    net_brake = output[0].detach().numpy()[2].item()
                else:
                    net_throttle = output.data.cpu().numpy()[0][0].item()
                    net_steer = output.data.cpu().numpy()[0][1].item()
                    net_brake = output.data.cpu().numpy()[0][2].item()
                control = carla.VehicleControl()
                control.throttle = net_throttle
                control.steer = net_steer
                control.brake = 0.0
                control.manual_gear_shift=True
                control.gear = 1
                #print(control)
                vehicle.apply_control(control) 

            elapsed_time = time.time() - init_time
            sleep_time = max(0, 1/frequency - elapsed_time)         
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':

    main()

    
