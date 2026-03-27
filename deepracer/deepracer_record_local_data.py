#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import signal
import argparse
import threading
import time

import cv2
import numpy as np
import csv

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from deepracer_interfaces_pkg.msg import ServoCtrlMsg


class DeepRacerRecorder(Node):
    """ROS 2 node that subscribes to ServoCtrlMsg and records images with telemetry data"""
    
    def __init__(self, dataset_path="./datasets", circuit="simple"):
        super().__init__('deepracer_recorder_node')
        
        self.dataset_path = dataset_path
        self.circuit = circuit
        self.iteration = 0
        self.stop_recording = False
        
        # Setup dataset directory
        self._setup_dataset_directory()
        
        # Initialize CSV writer
        self.csv_file = open(os.path.join(self.dataset_path, "data.csv"), "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['image_name', 'v', 'w'])
        self.csv_file.flush()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera device")
            raise RuntimeError("Cannot open camera")
        
        # Subscribe to ServoCtrlMsg
        self.subscription = self.create_subscription(
            ServoCtrlMsg,
            '/ctrl_pkg/servo_msg',
            self.servo_callback,
            10
        )
        self.get_logger().info(f"DeepRacer Recorder initialized. Saving to: {self.dataset_path}")
    
    def _setup_dataset_directory(self):
        """Create dataset directory structure"""
        # if the dataset directory is not present, then create it
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        
        # Create circuit-specific subdirectory
        circuit_dir_name = f"{self.circuit}_recorded_{len(os.listdir(self.dataset_path))}"
        circuit_path = os.path.join(self.dataset_path, circuit_dir_name)
        
        if not os.path.exists(circuit_path):
            os.mkdir(circuit_path)
        
        self.dataset_path = circuit_path
    
    def servo_callback(self, msg):
        """Callback when ServoCtrlMsg is received"""
        if self.stop_recording:
            return
        
        try:
            # Capture image
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning("Failed to capture frame from camera")
                return
            
            # Increment iteration and save image
            self.iteration += 1
            image_name = f"{self.iteration}.png"
            image_path = os.path.join(self.dataset_path, image_name)
            cv2.imwrite(image_path, frame)
            
            # Extract throttle and angle values from ServoCtrlMsg
            throttle = msg.throttle  # linear speed (v)
            angle = msg.angle        # angular speed (w)
            
            # Write to CSV
            self.csv_writer.writerow([image_name, throttle, angle])
            self.csv_file.flush()
            
            if self.iteration % 100 == 0:
                self.get_logger().info(f"Recorded {self.iteration} frames")
        
        except Exception as e:
            self.get_logger().error(f"Error in servo_callback: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_recording = True
        if self.csv_file:
            self.csv_file.close()
        if self.cap:
            self.cap.release()
        self.get_logger().info(f"Recording stopped. Total iterartions recorded: {self.iteration}")


def main():
    parser = argparse.ArgumentParser(description="DeepRacer Local Data Recorder")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets",
        help="Path to save datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="simple",
        help="Name of the circuit (default: simple)"
    )
    
    args = parser.parse_args()
    
    # Initialize ROS 2
    rclpy.init()
    
    # Create the recorder node
    recorder = DeepRacerRecorder(dataset_path=args.dataset_path, circuit=args.circuit)
    
    # Create executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(recorder)
    

    print("DeepRacer Recorder started. Press Ctrl+C to stop recording.")
    print(f"Saving data to: {recorder.dataset_path}")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        recorder.cleanup()
        executor.shutdown()
        recorder.destroy_node()


if __name__ == '__main__':
    main()
