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
import timm
from torchvision import transforms
from utils.pilotnet import PilotNet

import sys



try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

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

    parser.add_argument("--model", type=str, default='pilotnet', help="Model type")
    parser.add_argument("--saved_model_path", type=str, help="Path to the saved model")
    parser.add_argument("--town_name", type=str,default="Town01", help="Carla Map to load")
    parser.add_argument("--spawn_points_csv", type=str,default="./Town01_spawn_points.csv", help="File qith the spawn points of the CARLA map")
    parser.add_argument("--draw_spawn_points", type=bool,default=False, help="Enable or disable the visibility of the spawn points")
    parser.add_argument("--vehicle_name", type=str,default='vehicle.mercedes.coupe_2020', help="Car model to load")
    
    args = parser.parse_args()
    return args


def calculate_sides(hypotenuse, angle):
  """
  Calculates the two sides of a right triangle given the hypotenuse and an angle.

  Args:
    hypotenuse: The length of the hypotenuse of the triangle.
    represents the distance we need to be from the car
    angle: The angle of the triangle in degrees.
    represents the yaw angle of the car we need to be aligned with 

  Returns:
    A tuple containing the lengths of the two sides of the triangle.
    which are delta x and y
  """

  # Convert the angle to radians
  angle_radians = math.radians(angle)

  # Calculate the opposite side using the sine function
  opposite_side = hypotenuse * math.sin(angle_radians)

  # Calculate the adjacent side using the cosine function
  adjacent_side = hypotenuse * math.cos(angle_radians)

  return opposite_side, adjacent_side



def spectator_function(spectator, vehicle):
    try:
        metres_distance = 5
        vehicle_transform = vehicle.get_transform()
        y,x = calculate_sides(metres_distance, vehicle_transform.rotation.yaw )  
        spectator_pos = carla.Transform(vehicle_transform.location + carla.Location(x=-x,y=-y,z=5),
                                                carla.Rotation( yaw = vehicle_transform.rotation.yaw,pitch = -25))
        spectator.set_transform(spectator_pos)
    except:
        exit()
        

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
    state_dict = torch.load(args.saved_model_path,weights_only=True)
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
        model = timm.create_model('fastvit_mci0', pretrained=False)
        num_ftrs = model.head.classifier[-1].in_features
        model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
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
    
    args = parse_args()

    actor_list = []

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
        spawn_point_idx = random.randint(0, len(route_1_indices) - 1)

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
        
        control = carla.VehicleControl()

        while 1 :
            

            init_time = time.time()
            spectator_function(spectator, vehicle)
            image = camera_img[0]
            if image is not None:
                image = carla_to_rgb(image)
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
                vehicle.apply_control(control) 
            
            elapsed_time = time.time() - init_time
            model_frequency = 1.0 / elapsed_time
            print(f"Max Inference frequency: {model_frequency:.4f} Hz")
            #sys.stdout.write(f"\r Max Inference frequency: {model_frequency:.4f} Hz")
            #sys.stdout.flush()
            sleep_time = max(0, 1/frequency - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elapsed_time = time.time() - init_time
            #model_frequency = 1.0 / elapsed_time
            
            #sys.stdout.write(f"\r Max Inference frequency: {model_frequency:.4f} Hz")

            #sys.stdout.flush()
            


    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':

    main()

    
