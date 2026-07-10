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
    iteration = 1
    while(True): 
              
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
              
        # the 'q' button is set as the 
        # quitting button
        # and the 's' button is set as the 
        # saving button

        key = cv2.waitKey(25)
        
        if key == 113: #"q"
            break 
        if key == 115: #"s" 
            # Filename 
            filename = 'savedImage' + str(iteration) + '.jpg'
            iteration = iteration + 1
            # Using cv2.imwrite() method 
            # Saving the image 
            cv2.imwrite(filename, frame)
            print("saved image with name:" + filename)

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    
def calibrate_camera():
    CHECKERBOARD = (7,7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
     
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world spac
    imgpoints = [] # 2d points in image plane.
     
    images = glob.glob('*.jpg')
    counter = 0
    img_shape = [0,0]
    for fname in images:
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[0],CHECKERBOARD[1]), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                 
        # If found, add object points, image points (after refining them)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            img_shape = gray.shape[::-1]
            counter = counter + 1
            # Draw and display the corners
            """
            cv2.drawChessboardCorners(img, (7,7), fmname, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            """         
        #cv2.destroyAllWindows()

    N_imm = counter # number of calibration images
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        
    print("Found " + str(counter) + " valid images for calibration")
    print("img_shape=" + str(img_shape))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")


    #open and read the file after the overwriting:
    f = open("camera_distortion_info.txt", "w")
    f.write("Found " + str(counter) + " valid images for calibration")
    f.write("img_shape=" + str(img_shape))
    f.write("K=np.array(" + str(K.tolist()) + ")")
    f.write("D=np.array(" + str(D.tolist()) + ")")
    f.close()
    
    
    print("Saved information in file: camera_distortion_info.txt") 
    
    #View undistorted images
    """
    for path in glob.glob('*.jpg'):
        img = cv2.imread(path)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img_shape, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imshow('Original Image', img)
        cv2.imshow('Undistort Image', undistorted_img)
        cv2.waitKey(0)
     """

def main():

    if args.mode == "capture":
        capture_images()
        
    elif args.mode == "calibration":
        calibrate_camera()


# Execute!
if __name__ == "__main__":
    main()
