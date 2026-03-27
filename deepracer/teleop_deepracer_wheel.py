#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import signal
import select
import termios
import tty
import numpy as np
import time
import cv2
import utils.master_hal as HAL
import csv
import argparse
import threading
import subprocess
import pygame
import math
from configparser import ConfigParser
from utils.deepracer_metrics import DeepRacerMetrics

settings = termios.tcgetattr(sys.stdin)

Finish_program = False

THROTLE_VAL = 0.5
linear_speed = THROTLE_VAL
angular_speed = 0

# Camera and display globals for main thread
camera = None
display_surface = None
display_clock = None
resolution = None
metrics_process = None  # Subprocess for metrics recording

class GetWheelThread(threading.Thread):
    """Thread to read steering wheel inputs and update global speed variables.
    If the metrics are enabled, it also starts recording when the "manual_mode" is pressed (X button in our case)"""

    def __init__(self, wheel_config_path="./wheel_config.ini"):
        super().__init__()
        
        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please connect just one joystick")
        elif joystick_count == 0:
            raise RuntimeError("No joystick detected.")
        
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        
        # Load wheel configuration
        self._parser = ConfigParser()
        self._parser.read(wheel_config_path)
        self._steer_idx = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._manual_mode_idx = int(self._parser.get('G29 Racing Wheel', 'manual_mode'))

    def _parse_vehicle_wheel(self):
        """Parse steering wheel inputs and update global speed variables."""
        global linear_speed, angular_speed
        
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        
        # Steering mapping
        K1 = 1.0
        steerCmd = -K1 * math.tan(1.1 * jsInputs[self._steer_idx])
        angular_speed = steerCmd
        
        # Throttle mapping (accelerator pedal)
        K2 = 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1.0:
            throttleCmd = 1.0
        
        # Brake mapping (brake pedal)
        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1
        
        # Combine throttle and brake to linear speed
        # Positive for forward, negative for backward
        if brakeCmd > 0.1:  # If brake is applied
            linear_speed = -brakeCmd * (THROTLE_VAL - 0.1)
        else:
            linear_speed = throttleCmd * THROTLE_VAL

    def run(self):
        global Finish_program
        
        print("GetWheelThread started. Press button 0 (or 'q') to exit...")
        
        try:
            while not Finish_program:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.JOYBUTTONUP and event.button == 6):
                        print("Exit button pressed")
                        Finish_program = True
                        break
                
                # Update wheel inputs continuously
                if not Finish_program:
                    self._parse_vehicle_wheel()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
        
        except Exception as e:
            print(f"Error in GetWheelThread: {e}")
            Finish_program = True
        
        print("GetWheelThread exiting...")

class Brain:
    """Simple Brain class that sends linear and angular speed to the robot using the HAL methods"""

    def __init__(self, mode=None, circuit = "simple"):
        self.mode = mode
        self.circuit = circuit
        self.iteration = 0
        print(f"Brain initialized in {self.mode} mode, circuit: {self.circuit}")

    def execute(self):
        global linear_speed
        global angular_speed
        global Finish_program
        
        if not Finish_program:
            HAL.setV(round(linear_speed, 3))
            HAL.setW(round(angular_speed, 3))
            #print(f"Linear speed = {linear_speed:.3f}, Angular speed = {angular_speed:.3f}")
        else:
            time.sleep(0.001)
        
def end_signal_handler(sig, frame):
    print("\nShutting down...")
    cleanup()
    

def cleanup():
    global Finish_program, camera, display_surface, metrics_process
    Finish_program = True
    if camera is not None:
        camera.release()
    if display_surface is not None:
        pygame.quit()
    # Terminate metrics subprocess if it's still running
    if metrics_process is not None and metrics_process.poll() is None:
        metrics_process.terminate()
        try:
            metrics_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            metrics_process.kill()
            metrics_process.wait()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default=None, help="To save or not to save the data")
    parser.add_argument("--circuit", type=str, default="simple", help="Name of the circuit")
    parser.add_argument("--wheel_config", type=str, default="./wheel_config.ini", 
                        help="Path to wheel configuration file (default: ./wheel_config.ini)")
    parser.add_argument("--enable_metrics", action="store_true", help="Enable DeepRacerMetrics recording")
    parser.add_argument("--metrics_output_path", type=str, default="output.avi", 
                        help="Path to save the metrics video (default: output.avi)")
    parser.add_argument("--metrics_csv_path", type=str, default="./centroid_data.csv", 
                        help="Path to save centroid data CSV (default: centroid_data.csv)")
    parser.add_argument("--metrics_checkpoints_path", type=str, default=None, 
                        help="Path to checkpoints CSV for trajectory comparison")

    args = parser.parse_args()
    return args

args = parse_args()
brain = Brain(args.mode, args.circuit)

def user_main():
    global Finish_program
    global camera, display_surface, display_clock, resolution

    # Display camera feed if camera is available
    if camera is not None:
        ret, frame = camera.read()
        if ret:
            # Convert BGR to RGB for pygame display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Transpose for pygame display
            rgb_frame = np.transpose(rgb_frame, (1, 0, 2))
            # Create surface and blit to display
            surf = pygame.surfarray.make_surface(rgb_frame)
            display_surface.blit(surf, (0, 0))
            pygame.display.update()
            display_clock.tick(30)
    
    brain.execute()
    if Finish_program:
        print("Finishing program...")
        HAL.setV(0)
        HAL.setW(0)
        cleanup()
        sys.exit(0)


def main():

    # Initialize pygame for joystick support
    pygame.init()
    
    # Global references
    global camera, display_surface, display_clock, resolution, metrics_process
    
    # Initialize camera for display in main thread (only if metrics disabled)
    try:
        if not args.enable_metrics:
            camera = cv2.VideoCapture(2)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    h, w = frame.shape[:2]
                    resolution = (w, h)
                    display_surface = pygame.display.set_mode(resolution)
                    pygame.display.set_caption("DeepRacer - Main Camera Feed")
                    display_clock = pygame.time.Clock()
                    print(f"Camera initialized: {w}x{h}")
                else:
                    print("Warning: Could not read first frame from camera")
                    camera = None
            else:
                print("Warning: Could not open camera device 0")
                camera = None
        else:
            print("Metrics enabled - camera display disabled in main thread")
    except Exception as e:
        print(f"Warning: Error initializing camera: {e}")
        camera = None
    
    # Create and start the joystick thread
    key_thread = GetWheelThread(args.wheel_config)
    key_thread.daemon = True
    key_thread.start()
    
    # Start metrics as a subprocess if enabled
    if args.enable_metrics:
        try:
            cmd = [
                sys.executable, "-m", "utils.deepracer_metrics",
                "--output_path", args.metrics_output_path,
                "--csv_path", args.metrics_csv_path
            ]
            if args.metrics_checkpoints_path:
                cmd.extend(["--checkpoints_path", args.metrics_checkpoints_path])
            
            metrics_process = subprocess.Popen(cmd)
            print("DeepRacer Metrics recording subprocess started")
        except Exception as e:
            print(f"Error starting DeepRacer Metrics subprocess: {e}")
    
    signal.signal(signal.SIGINT, end_signal_handler)
    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)
    
# Execute!
if __name__ == "__main__":
    main()
