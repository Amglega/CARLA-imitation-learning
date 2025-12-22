#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--mode", type=str, default="capture", help="To save or not to save the data")


    args = parser.parse_args()
    return args

args = parse_args()

def capture_images():

    vid = cv2.VideoCapture(0)
    img_shape=(640, 480)
    K=np.array([[502.4827132565883, 0.0, 320.49002418357725], [0.0, 502.4546524395416, 238.255941996664], [0.0, 0.0, 1.0]])
    D=np.array([[-0.08703736838521056], [-0.2917938213212864], [0.6776229437062419], [-0.3476415479534463]])

    while(True): 
              
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img_shape, cv2.CV_16SC2)
        undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        #cv2.imshow('Original Image', frame)
        # Display the resulting frame 
        cv2.imshow('Undistort Image', undistorted_img)
        
        # the 'q' button is set as the 
        # quitting button

        key = cv2.waitKey(25)
        
        if key == 113: #"q"
            break 

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def main():
    capture_images()
        


# Execute!
if __name__ == "__main__":
    main()
