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


def create_dataset_directory(path,dir_name="CARLA_dataset"):
  i = 0
  while True:
    directory_name = f"{path}/{dir_name}_{i}"
    if not os.path.exists(directory_name):
      os.makedirs(directory_name)
      print(f"Created directory: {directory_name}")
      return directory_name
    i += 1


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
        client.set_timeout(10.0)

        # Once we have a client we can retrieve the world that is currently
        # running.

        client.load_world_if_different('Town01')
        world = client.get_world()

        weather = Weather(world.get_weather())

        weather.set_weather_atributes(clouds=20, sun_azimuth=210, sun_altitude=20)
        world.set_weather(weather.weather)
               
        spawn_points = world.get_map().get_spawn_points()

        # Draw the spawn point locations as numbers in the map
        print("Drawing spawn points")
        for i, spawn_point in enumerate(spawn_points):
            world.debug.draw_string(spawn_point.location, str(i), life_time=10000)
        print("Starting infinite loop")
        # In synchronous mode, we need to run the simulation to fly the spectator
        while True:
            world.tick()
    except KeyboardInterrupt:
        print('Done')
        
    


if __name__ == '__main__':

    main()

    
