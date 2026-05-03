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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile
from deepracer_interfaces_pkg.msg import ServoCtrlMsg
from std_msgs.msg import Bool


class DeepRacerRecorder(Node):
    """ROS 2 node that subscribes to ServoCtrlMsg and records images with telemetry data"""
    
    def __init__(self, dataset_path="./datasets", circuit="simple", camera=None):
        super().__init__('deepracer_recorder_node')
        
        self.dataset_path = dataset_path
        self.circuit = circuit
        self.iteration = 0
        self.stop_recording = True
        
        # Setup dataset directory
        self._setup_dataset_directory()
        
        # Initialize CSV writer
        self.csv_file = open(os.path.join(self.dataset_path, "data.csv"), "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['image_name', 'v', 'w'])
        self.csv_file.flush()
        
        # Use shared camera or create a new one
        if camera is not None:
            self.cap = camera
            self.owns_camera = False
        else:
            self.cap = cv2.VideoCapture(0)
            self.owns_camera = True
            if not self.cap.isOpened():
                self.get_logger().error("Failed to open camera device")
                raise RuntimeError("Cannot open camera")
        
        # Subscribe to ServoCtrlMsg
        servo_cb_group = MutuallyExclusiveCallbackGroup()
        
        self.subscription = self.create_subscription(
            ServoCtrlMsg,
            '/ctrl_pkg/servo_msg',
            self.servo_callback, 
            callback_group = servo_cb_group,
            qos_profile = QoSProfile(depth=10)
        )

        # Subscribe to start_record Bool topic
        self.start_record_subscription = self.create_subscription(
            Bool,
            '/start_record',
            self.start_record_callback,
            callback_group = None,
            qos_profile = QoSProfile(depth=10)
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
    
    def start_record_callback(self, msg):
        """Callback when Bool message is received"""
        self.stop_recording = not msg.data
        if msg.data:
            self.get_logger().info("Recording started")
        else:
            self.get_logger().info("Recording stopped")
    
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
        # Only release camera if this instance created it
        if self.cap and self.owns_camera:
            self.cap.release()
        self.get_logger().info(f"Recording stopped. Total iterartions recorded: {self.iteration}")


def display_camera_feed(cap, stop_event):
    """Display camera feed in a separate thread"""
    window_name = "Camera Feed - Press 'q' to quit"
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break
            
            cv2.imshow(window_name, frame)
            
            # Check for 'q' key press (waitKey needs at least 1ms)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("Camera display stopped by user")
                stop_event.set()
                break
    finally:
        cv2.destroyAllWindows()


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
        default="simple_wheel",
        help="Name of the circuit (default: simple)"
    )
    
    args = parser.parse_args()
    
    # Initialize camera in main
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open camera device")
        return
    
    # Initialize ROS 2
    rclpy.init()
    
    # Create the recorder node with shared camera
    recorder = DeepRacerRecorder(dataset_path=args.dataset_path, circuit=args.circuit, camera=cap)
    
    # Create executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(recorder)
    
    # Create event to signal camera display thread to stop
    stop_event = threading.Event()
    
    # Start camera display thread
    display_thread = threading.Thread(target=display_camera_feed, args=(cap, stop_event), daemon=True)
    display_thread.start()
    
    print("DeepRacer Recorder started. Press Ctrl+C to stop recording or 'q' in the camera window.")
    print(f"Saving data to: {recorder.dataset_path}")
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        stop_event.set()
        recorder.cleanup()
        cap.release()
        executor.shutdown()
        recorder.destroy_node()
        display_thread.join(timeout=2)


if __name__ == '__main__':
    main()
