import cv2 as cv
import pygame
import numpy as np
import csv
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox)
from PyQt5.QtGui import QFont


class MarkerClickCapture:
    """
    Captures marker points by clicking on a pygame camera display.
    """
    def __init__(self, camera_index=2):
        self.camera_index = camera_index
        self.cap = cv.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Get first frame to determine resolution
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        
        h, w = frame.shape[:2]
        self.resolution = (w, h)
        self.markers = []
        self._current_frame = frame
        
        # Initialize pygame
        pygame.init()
        self.display = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("Click on 4 markers (ESC to cancel)")
        
        # Cursor tracking
        pygame.mouse.set_visible(True)
    
    def capture_markers(self):
        """
        Display camera feed and capture 4 marker clicks.
        Returns list of (x, y) tuples or None if cancelled.
        """
        clock = pygame.time.Clock()
        
        while len(self.markers) < 4:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self._current_frame = frame
            
            # Display frame with markers
            display_frame = frame.copy()
            
            # Draw existing markers
            for i, (x, y) in enumerate(self.markers):
                cv.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
                cv.putText(display_frame, str(i + 1), (x + 15, y + 15), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instruction text
            text = f"Click on marker {len(self.markers) + 1} of 4"
            cv.putText(display_frame, text, (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convert and display on pygame
            rgb_frame = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            rgb_frame = np.transpose(rgb_frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(rgb_frame)
            self.display.blit(surf, (0, 0))
            pygame.display.update()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    x, y = pygame.mouse.get_pos()
                    self.markers.append((x, y))
                    print(f"Marker {len(self.markers)} captured at ({x}, {y})")
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        self.cap.release()
                        return None
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    self.cap.release()
                    return None
            
            clock.tick(30)
        
        pygame.quit()
        self.cap.release()
        return self.markers


class RealWorldCoordinatesInput(QWidget):
    """
    PyQt5 dialog to input real-world coordinates for captured markers.
    """
    def __init__(self, image_coords, output_csv_path="markers.csv"):
        super().__init__()
        self.image_coords = image_coords
        self.output_csv_path = output_csv_path
        self.real_coords = [None] * len(image_coords)
        self.line_edits = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Enter Real-World Coordinates")
        self.setGeometry(100, 100, 600, 400)
        
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Enter real-world coordinates for each marker:")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Grid for marker coordinates
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel("Marker ID"), 0, 0)
        grid_layout.addWidget(QLabel("Image Coords"), 0, 1)
        grid_layout.addWidget(QLabel("Real X"), 0, 2)
        grid_layout.addWidget(QLabel("Real Y"), 0, 3)
        
        for i, (img_x, img_y) in enumerate(self.image_coords):
            # Marker ID
            marker_id_label = QLabel(str(i + 1))
            grid_layout.addWidget(marker_id_label, i + 1, 0)
            
            # Image coordinates (read-only)
            img_coords_label = QLabel(f"({img_x}, {img_y})")
            grid_layout.addWidget(img_coords_label, i + 1, 1)
            
            # Real X input
            real_x_input = QLineEdit()
            real_x_input.setPlaceholderText("Real X")
            grid_layout.addWidget(real_x_input, i + 1, 2)
            self.line_edits.append(("real_x", i, real_x_input))
            
            # Real Y input
            real_y_input = QLineEdit()
            real_y_input.setPlaceholderText("Real Y")
            grid_layout.addWidget(real_y_input, i + 1, 3)
            self.line_edits.append(("real_y", i, real_y_input))
        
        main_layout.addLayout(grid_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save CSV")
        save_button.clicked.connect(self.save_csv)
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close_app)
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def save_csv(self):
        """Save the marker data to CSV."""
        # Extract real-world coordinates from input fields
        real_coords = [None] * len(self.image_coords)
        
        for field_type, marker_idx, line_edit in self.line_edits:
            try:
                value = float(line_edit.text())
                if real_coords[marker_idx] is None:
                    real_coords[marker_idx] = [None, None]
                
                if field_type == "real_x":
                    real_coords[marker_idx][0] = value
                else:  # real_y
                    real_coords[marker_idx][1] = value
            except ValueError:
                QMessageBox.warning(self, "Input Error", 
                                  f"Invalid input in marker {marker_idx + 1}. Please enter valid numbers.")
                return
        
        # Check if all coordinates are filled
        for i, coords in enumerate(real_coords):
            if coords is None or coords[0] is None or coords[1] is None:
                QMessageBox.warning(self, "Input Error", 
                                  f"Please fill in all coordinates for marker {i + 1}.")
                return
        
        # Write to CSV
        try:
            with open(self.output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['marker_id', 'image_x', 'image_y', 'real_x', 'real_y'])
                
                for i, (img_x, img_y) in enumerate(self.image_coords):
                    real_x, real_y = real_coords[i]
                    writer.writerow([i + 1, img_x, img_y, int(real_x), int(real_y)])
            
            QMessageBox.information(self, "Success", 
                                   f"Markers saved to {self.output_csv_path}")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV: {str(e)}")
    
    def close_app(self):
        """Close the application."""
        self.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create markers CSV for perspective transformation")
    parser.add_argument("--output", type=str, default="markers.csv", 
                       help="Output CSV file path")
    parser.add_argument("--camera", type=int, default=2, 
                       help="Camera index to use")
    args = parser.parse_args()
    
    print("Starting marker capture...")
    print("Instructions: Click on 4 points in the camera feed")
    
    try:
        # Get marker coords from camera clicks
        capturer = MarkerClickCapture(camera_index=args.camera)
        markers = capturer.capture_markers()
        
        if markers is None:
            print("Marker capture cancelled.")
            sys.exit(0)
        
        print(f"Captured {len(markers)} markers: {markers}")
        
        # Get real-world coordinates via PyQt5
        app = QApplication(sys.argv)
        dialog = RealWorldCoordinatesInput(markers, output_csv_path=args.output)
        dialog.show()
        sys.exit(app.exec_())
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
