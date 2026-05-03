import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define original points, defined by camera perspective [x, y]
src_pts = np.float32([
    [617,441], # Bottom right marker
    [90,274], # Bottom left marker
    [468,62], # Upper right marker
    [150,80]  # Upper left marker
])

# Define real-world coordinates [x, y]
dst_pts = np.float32([
    [58,58], # Bottom right marker
    [174,348], # Bottom left marker
    [580,58],  # Upper right marker
    [522,406] # Upper left marker
])

# Get the Transformation Matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
print("Perspective Transformation Matrix:\n", M.shape)
try:
    # Define output image size (width, height)
    width, height = 650, 500

    video_path = 'output.avi'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    warped_frame = cv2.warpPerspective(frame, M, (width, height))
    
    # Convert BGR to RGB for display
    warped_rgb = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.subplot(121), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(122), plt.imshow(warped_rgb), plt.title('Transformed View')
    plt.show()
except NameError:
    print('Error: "frame" not found.')