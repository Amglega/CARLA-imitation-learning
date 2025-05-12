#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import threading
import numpy as np
import random
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import carla
import random
import time
from datetime import datetime
import cv2 as cv
from rclpy.executors import MultiThreadedExecutor
import csv
import argparse
import math
# Import a basic agent
from agents.navigation.basic_agent import BasicAgent

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



class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """
    def __init__(self):
    # Initiate the Node class's constructor and give it a name
        super().__init__('image_subscriber')
        
        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
        Image, 
        '/carla/autopilot/front/image', 
        self.listener_callback, 
        10)
        self.subscription # prevent unused variable warning
        self.image = np.zeros((3, 3, 3), np.uint8)
        self.lock = threading.Lock()
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def listener_callback(self, data):
        # Display the message on the console
        #self.get_logger().info('Receiving video frame')
        self.lock.acquire()
        # Convert ROS Image message to OpenCV image
        self.image = self.br.imgmsg_to_cv2(data,"bgr8")
        self.lock.release()

    def getImage(self):
        self.lock.acquire()
        return_img = self.image
        self.lock.release()
        return return_img


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


def main():
    
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
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.

        client.load_world_if_different('Town01')
        world = client.get_world()

        weather = Weather(world.get_weather())

        weather.set_weather_atributes(clouds=20, sun_azimuth=210, sun_altitude=20)
        world.set_weather(weather.weather)
               

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        # 
        bp = blueprint_library.find('vehicle.mercedes.coupe_2020')

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Create route from the chosen spawn points
        spawn_points = world.get_map().get_spawn_points()
        route_1_indices = load_spawn_points("./Town01_spawn_points.csv")

        # Draw spawn points
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

        # Let's add now a "rgb" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)



        # Path to save the datasets
        current_path = os.getcwd() + "/" + "datasets"
        dataset_path = create_dataset_directory(current_path)

        writer_output = csv.writer(open(dataset_path + "/data.csv", "w"))
            
        writer_output.writerow(["image_name","throttle","steer","brake"])
        
        # Frecuency at which the images and car commands will be saved
        frequency = 60
        time_cycle = 1000.0 / frequency
        
        spectator = world.get_spectator()
        spectator.set_transform(get_transform(vehicle.get_location(), -90))

        # To start a basic agent
        agent = BasicAgent(vehicle)
        destination_idx = spawn_point_idx+1
        if destination_idx > (len(route_1_indices)-1):
            destination_idx = 0
        destination = spawn_points[route_1_indices[destination_idx]].location
        agent.set_destination(destination)
    
        random_mode = False
        random_mode_period = 10
        change_to_auto_time = 1
        next_random_mode_time = time.time() + random_mode_period
        random_timer_end = next_random_mode_time + change_to_auto_time
        prev_time = time.time()

        finish = False
        
        while rclpy.ok() or not finish:
            start_time = datetime.now()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                done = True     

            if agent.done():
                destination_idx += 1
                if destination_idx > (len(route_1_indices)-1):
                    destination_idx = 0
                destination = spawn_points[route_1_indices[destination_idx]].location
                agent.set_destination(destination)
            
            if not random_mode:
                control = agent.run_step()
                vehicle.apply_control(control)
                executor.spin_once()
                image = camera_listener.getImage()
                if image.shape[0] > 50:
                    iteration+=1
                    cv.imwrite(dataset_path + "/" + str(iteration) + ".png", image)
                    writer_output.writerow([str(iteration) + '.png', control.throttle, control.steer, control.brake])          
            else:
                control = vehicle.get_control()
                control.steer = random.uniform(-0.20, 0.20)  #0.2 steer y 0.1 throttle
                control.throttle = 0.1
                vehicle.apply_control(control)
            
            
            # Change to random control mode during 1 second
            if time.time() >= next_random_mode_time:
                random_mode = True
                next_random_mode_time = time.time() + random_mode_period 
                random_timer_end =  time.time() + change_to_auto_time
                print("CAMBIO A RANDOM")

            if  random_mode and time.time() > random_timer_end:
                random_mode = False
                print("CAMBIO A AUTO")
            
            finish_time = datetime.now()
            dt = finish_time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if(ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)
            

    finally:
    
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        rclpy.shutdown()
        print('done.')

        
    


if __name__ == '__main__':

    main()

    
