#!/usr/bin/python
# -*- coding: utf-8 -*-

# Script based on the poyect Behaviour Metrics, it captures video from the camera, 
# detects the car (using color detection), calculates the centroid, and records both the video and centroid data. 
# It also provides functionality to compare the recorded trajectory 
# with reference checkpoints form past iterations and visualize the results in a wrapped perspective view.

import argparse
import math
import threading
import cv2 as cv
import pygame
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

class DeepRacerMetrics:
    def __init__(self, output_path="output.avi", fps=60.0, csv_path="centroid_data.csv", checkpoints_path=None, markers_csv_path=None):
        self.output_path = output_path
        self.fps = fps
        self.csv_path = csv_path
        self.checkpoints_path = checkpoints_path
        self.markers_csv_path = markers_csv_path
        self.csv_file = None
        self.csv_writer = None
        self.fourcc = cv.VideoWriter_fourcc(*'XVID')
        
        # Thread-safe event for recording toggle from keyboard in standalone display mode
        self._record_toggle_event = threading.Event()
        
        # If camera is provided, use it; otherwise open a new one

        self.cap = cv.VideoCapture(2)
            
            
        self.recording = False
        self.waypoints = []
        # get the resolution from first frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("cannot read from camera")
        h, w = frame.shape[:2]
        self.resolution = (w, h)
        self.writer = cv.VideoWriter(self.output_path, self.fourcc, self.fps, self.resolution)
        # put the frame back for display loop
        self._last_frame = frame

        pygame.init()
        self.display = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("DeepRacer Camera")

        

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            # initialize CSV file and writer
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # write header
            self.csv_writer.writerow(['x', 'y'])
            self.csv_file.flush()
            print("started recording")
        else:
            self.recording = False
            self.writer.release()
            # close CSV file
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            print("stopped recording")
            if self.checkpoints_path is not None and len(self.waypoints) > 0:
                if os.path.exists(self.checkpoints_path):
                    self.compare_trajectory()

    def get_position_deviation_and_effective_completed_distance(self):
        # Placeholder for actual track center and distance calculations
        f = open(self.checkpoints_path, 'r')
        reader = csv.reader(f)
        next(reader) # skip header x,y
        checkpoints = [(int(row[0]), int(row[1])) for row in reader]   
        min_dists, best_checkpoint_points_x, best_checkpoint_points_y = [], [], []
        covered_checkpoints = []
        for cx, cy in checkpoints:
            best_d = float("inf")
            bx, by = cx, cy
            for mx, my in self.waypoints:
                d = math.hypot(mx - cx, my - cy)
                print(d)
                if d < best_d:
                    best_d = d
                    bx, by = mx, my
            best_checkpoint_points_x.append(bx)
            best_checkpoint_points_y.append(by)
            if best_d < 200.0:
                min_dists.append(best_d)
                if (not covered_checkpoints or covered_checkpoints[-1] != (bx, by)) and (best_d < 1.0):
                    covered_checkpoints.append((bx, by))
        f.close()
        return min_dists, best_checkpoint_points_x, best_checkpoint_points_y, covered_checkpoints

    def display_wrapped_trajectories(self, waypoints, checkpoints):
        """
        Display both trajectories (recorded and reference) in a wrapped perspective view.
        Applies perspective transformation to visualize trajectories from a top-down view.
        
        Args:
            waypoints: List of (x, y) tuples representing recorded trajectory centroids
            checkpoints: List of (x, y) tuples representing reference trajectory checkpoints
        """
        # Read marker points from CSV file if available
        src_pts, dst_pts = self._load_perspective_points()
        
        if src_pts is None or dst_pts is None:
            print("Warning: Could not load perspective points from markers CSV. Using default points.")
            # Define default source points (camera perspective) [x, y]
            src_pts = np.array([
                [617, 441],  # Bottom right marker
                [90, 274],   # Bottom left marker
                [468, 62],   # Upper right marker
                [150, 80]    # Upper left marker
            ], dtype=np.float32)
            
            # Define default real-world coordinates [x, y]
            dst_pts = np.array([
                [58, 58],     # Bottom right marker
                [174, 348],   # Bottom left marker
                [580, 58],    # Upper right marker
                [522, 406]    # Upper left marker
            ], dtype=np.float32)
        
        # Calculate the Perspective Transformation Matrix
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        
        # Define output image size (width, height)
        width, height = 650, 500
        
        # Try to get a frame for the background
        frame = None
        if hasattr(self, '_last_frame'):
            frame = self._last_frame
        else:
            # Try to read from the video file or camera
            ret, frame = self.cap.read()
            if not ret:
                # Create a blank frame if unable to read
                frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Apply perspective warp to the frame
        warped_frame = cv.warpPerspective(frame, M, (width, height))
        warped_rgb = cv.cvtColor(warped_frame, cv.COLOR_BGR2RGB)
        
        # Prepare waypoints for transformation
        if waypoints:
            waypoints_array = np.array(waypoints, dtype=np.float32).reshape(-1, 1, 2)
            transformed_waypoints = cv.perspectiveTransform(waypoints_array, M).reshape(-1, 2)
        else:
            transformed_waypoints = np.array([])
        
        # Prepare checkpoints for transformation
        if checkpoints:
            checkpoints_array = np.array(checkpoints, dtype=np.float32).reshape(-1, 1, 2)
            transformed_checkpoints = cv.perspectiveTransform(checkpoints_array, M).reshape(-1, 2)
        else:
            transformed_checkpoints = np.array([])
        
        # Create figure for wrapped perspective view
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display the warped frame as background
        ax.imshow(warped_rgb)
        
        # Plot transformed trajectories
        if len(transformed_waypoints) > 0:
            ax.plot(transformed_waypoints[:, 0], transformed_waypoints[:, 1], 'b-o', 
                   label='Recorded Trajectory (Centroids)', linewidth=2, markersize=4)
        
        if len(transformed_checkpoints) > 0:
            ax.plot(transformed_checkpoints[:, 0], transformed_checkpoints[:, 1], 'r-s', 
                   label='Reference Trajectory (Checkpoints)', linewidth=2, markersize=4)
        
        ax.set_xlabel('X Coordinate (World Frame)')
        ax.set_ylabel('Y Coordinate (World Frame)')
        ax.set_title('Trajectories in Wrapped Perspective View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _load_perspective_points(self):
        """
        Load perspective transformation points from a CSV file.
        
        CSV format: marker_id, image_x, image_y, real_x, real_y
        
        Returns:
            (src_pts, dst_pts) as numpy arrays, or (None, None) if file not found or invalid
        """
        if self.markers_csv_path is None or not os.path.exists(self.markers_csv_path):
            return None, None
        
        try:
            src_pts_list = []
            dst_pts_list = []
            
            with open(self.markers_csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) < 5:
                        continue
                    marker_id = int(row[0])
                    image_x = float(row[1])
                    image_y = float(row[2])
                    real_x = float(row[3])
                    real_y = float(row[4])
                    
                    src_pts_list.append([image_x, image_y])
                    dst_pts_list.append([real_x, real_y])
            
            if len(src_pts_list) > 0 and len(dst_pts_list) > 0:
                src_pts = np.array(src_pts_list, dtype=np.float32)
                dst_pts = np.array(dst_pts_list, dtype=np.float32)
                return src_pts, dst_pts
            else:
                return None, None
        except Exception as e:
            print(f"Error loading perspective points: {e}")
            return None, None

    def compare_trajectory(self):
        # Get position deviations and checkpoint points
        min_dists, best_checkpoint_points_x, best_checkpoint_points_y, covered_checkpoints = \
            self.get_position_deviation_and_effective_completed_distance()
        # Calculate total and mean deviation
        total_deviation = sum(min_dists) if min_dists else 0
        mean_deviation = total_deviation / len(min_dists) if min_dists else 0
        
        # Extract waypoints and checkpoints for plotting
        waypoints_x = [wp[0] for wp in self.waypoints]
        waypoints_y = [wp[1] for wp in self.waypoints]
        
        checkpoints = []
        if self.checkpoints_path:
            f = open(self.checkpoints_path, 'r')
            reader = csv.reader(f)
            next(reader) # skip header x,y
            checkpoints = [(int(row[0]), int(row[1])) for row in reader]
            f.close()
        
        checkpoints_x = [cp[0] for cp in checkpoints]
        checkpoints_y = [cp[1] for cp in checkpoints]
        
        # Create matplotlib figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Both trajectories overlaid
        ax1.plot(waypoints_x, waypoints_y, 'b-o', label='Recorded Trajectory (Centroids)', linewidth=2, markersize=4)
        ax1.plot(checkpoints_x, checkpoints_y, 'r-s', label='Reference Trajectory (Checkpoints)', linewidth=2, markersize=4)
        
        # Draw deviation lines between closest points
        #for i, (cx, cy) in enumerate(checkpoints):
        #    ax1.plot([cx, best_checkpoint_points_x[i]], [cy, best_checkpoint_points_y[i]], 
        #            'g--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviation distances
        ax2.bar(range(len(min_dists)), min_dists, color='orange', alpha=0.7)
        ax2.axhline(y=mean_deviation, color='r', linestyle='--', linewidth=2, label=f'Mean Deviation: {mean_deviation:.2f}')
        ax2.set_xlabel('Checkpoint Index')
        ax2.set_ylabel('Deviation Distance (pixels)')
        ax2.set_title('Position Deviations per Checkpoint')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Display trajectories in wrapped perspective view
        self.display_wrapped_trajectories(self.waypoints, checkpoints)
        
        # Print statistics
        print("\n" + "="*50)
        print("TRAJECTORY COMPARISON RESULTS")
        print("="*50)
        print(f"Total Deviation: {total_deviation:.2f} pixels")
        print(f"Mean Deviation: {mean_deviation:.2f} pixels")
        print(f"Number of Checkpoints: {len(min_dists)}")
        if min_dists:
            print(f"Max Deviation: {max(min_dists):.2f} pixels")
            print(f"Min Deviation: {min(min_dists):.2f} pixels")
        print(f"Covered Checkpoints: {len(covered_checkpoints)}/{len(checkpoints)}")
        print("="*50 + "\n")

    def run(self):
        clock = pygame.time.Clock()
        
        while True:
            
            if hasattr(self, "_last_frame"):
                frame = self._last_frame
                del self._last_frame
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break
            # apply red color mask
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # range for red color in HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # masks for both red ranges
            mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
            mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
            mask = cv.bitwise_or(mask1, mask2)
            
            # apply mask
            red_objects = cv.bitwise_and(frame, frame, mask=mask)
            
            # calculate centroid of red objects
            M = cv.moments(mask)
            cx, cy = None, None
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # change array to display correclty in pygame
            if self.display is not None:
                # change RGB to BGR and display on pygame window
                cv.circle(red_objects,(cx,cy),20,color=(255,255,255),thickness=3)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                surf = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
                self.display.blit(surf, (0, 0))
                pygame.display.update()

            if self.recording:
                self.writer.write(frame)
                # Write centroid coordinates to CSV if it exists
                if cx is not None and cy is not None:
                    self.waypoints.append((cx, cy))
                    self.csv_writer.writerow([cx, cy])
                    self.csv_file.flush()

            # Handle pygame events
            
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_r:
                        self.toggle_recording()
                    elif event.key == pygame.K_ESCAPE:
                        self.cleanup()
                        return
                elif event.type == pygame.QUIT:
                    self.cleanup()
                    return
            
            # Check for external recording toggle signal (from keyboard)
            if self._record_toggle_event.is_set():
                self.toggle_recording()
                self._record_toggle_event.clear()
            
            clock.tick_busy_loop(self.fps)

        self.cleanup()

    def cleanup(self):
        if self.recording:
            self.writer.release()
            if self.csv_file:
                self.csv_file.close()
        # Only release camera if we own it (didn't receive it from caller)
        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepRacer Metrics Recorder")
    parser.add_argument("--output_path", type=str, default="output.avi", help="Path to save the video")
    parser.add_argument("--csv_path", type=str, default="centroid_data.csv", help="Path to save the centroid data CSV")
    parser.add_argument("--checkpoints_path", type=str, default=None, help="Path to the checkpoints CSV for trajectory comparison")
    parser.add_argument("--markers_csv_path", type=str, default=None, help="Path to the markers CSV for perspective transformation points")
    args = parser.parse_args()
    recorder = DeepRacerMetrics(output_path=args.output_path, csv_path=args.csv_path, checkpoints_path=args.checkpoints_path, markers_csv_path=args.markers_csv_path)
    recorder.run()
