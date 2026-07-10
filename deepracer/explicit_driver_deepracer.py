#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import cv2
import utils.hal as HAL
import csv
import argparse

class Brain:

    def __init__(self, mode=None, circuit = "simple"):
        self.x_middle_left_above = 0
        self.deviation_left = 0
        self.iteration = 0
        self.circuit = circuit 
        self.path = os.getcwd() + "/" + "datasets"
        self.mode = mode
        #self.vid = cv2.VideoCapture(0)
        if mode == "save":
             # if the dataset directory is not present  
            # then create it. 
            if not os.path.exists(self.path): 
                os.mkdir(self.path)
            if not os.path.exists(self.path + "/" + self.circuit +"_teleop_"+ str(len(os.listdir(self.path)))):
                number = len(os.listdir(self.path))
                os.mkdir(self.path + "/" + self.circuit + "_teleop_" + str(number))
                self.path = self.path + "/" + self.circuit + "_teleop_" + str(number)
            else:
                self.path = self.path + "/" + self.circuit + "_teleop_" + str(len(os.listdir(self.path)))

            self.writer_output = csv.writer(open(self.path + "/data.csv", "w"))
            self.writer_output.writerow(['image_name','v','w'])
        else:
            self.writer_output = None



    def straight_case(self, deviation):
        if abs(deviation) < 35:
            rotation = (0.0035 * deviation + 0.05 * (deviation - self.deviation_left))

        elif abs(deviation) < 10:
            rotation = (0.004 * deviation + 0.05 * (deviation - self.deviation_left))

        else:
            rotation = (0.009 * deviation + 0.05 * (deviation - self.deviation_left))
            
        speed = 0.53
        return speed, rotation

    def curve_case(self, deviation):
        if abs(deviation) < 50:
            rotation = (0.001 * deviation + 0.0005 * (deviation - self.deviation_left))

        elif abs(deviation) < 80:
            rotation = (0.0011 * deviation + 0.0005 * (deviation - self.deviation_left))
 
        elif abs(deviation) < 130:
            rotation = (0.0012 * deviation + 0.0005 * (deviation - self.deviation_left))
  
        elif abs(deviation) < 190:
            rotation = (0.0012 * deviation + 0.0006 * (deviation - self.deviation_left))
            
        else:
            rotation = (0.0013 * deviation + 0.0006 * (deviation - self.deviation_left))
   
        speed = 0.53
        return speed, rotation

    def execute(self):
        
        image = HAL.getImage()
        #ret, image = self.vid.read()

        
        if image.shape[0] > 20:

            deviation = 0
            if self.mode == "save":
                self.iteration += 1
                cv2.imwrite(self.path + "/" + str(self.iteration) + ".png", image)


            image_cropped = image[230:, :, :]
            
            image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
            # red mask
            lower_red = np.array([160,50, 50])
            upper_red = np.array([180,255,255])
            image_mask = cv2.inRange(image_hsv, lower_red, upper_red)
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            filtered_img = cv2.erode(image_mask, erosion_kernel, iterations = 1)

            # calculate moments of binary image
            """
            for i in range(filtered_img.shape[0]-1):
                for j in range((filtered_img.shape[1]*1)//3,(filtered_img.shape[1]*2)//3):
                    filtered_img[i,j] = 0
            """
            M = cv2.moments(filtered_img)
            pixel_count= np.sum(filtered_img==255)

            #(str(pixel_count))
                
            if M["m00"] !=0 and pixel_count > 100:
            
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # put text and highlight the center
                cv2.circle(filtered_img, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(filtered_img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                deviation = (filtered_img.shape[1]/2)-cX + 160

                if abs(deviation) < 30:
                    speed, rotation = self.straight_case(deviation)
                    print("Recta")
                else:
                    speed, rotation = self.curve_case(deviation)
                    print("Curva")
                #print(deviation)
                self.deviation_left = deviation
            else:
            
                if self.deviation_left > 0:
                    rotation = -1.0
                else:
                    rotation = 1.0
                speed = -0.4
                print("OUT of Range")
                
           
            if self.mode == "debug":              
                cv2.imshow('Undistort Image', filtered_img)
                key = cv2.waitKey(1)
                print("V:" + str(speed) + ",W:"+ str(rotation))
                print("Deviation: " + str(deviation))
                HAL.setV(speed)
                HAL.setW(rotation)
            else:
                HAL.setV(speed)
                HAL.setW(rotation)
                print("V:" + str(speed) + ",W:"+ str(rotation))
            if self.mode == "save":
                self.writer_output.writerow([str(self.iteration) + '.png',speed,rotation])
        else:
            time.sleep(1)
        
	
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, help="To save or not to save the data")
    parser.add_argument("--circuit", type=str, default="simple", help="Name of the circuit")

    args = parser.parse_args()
    return args

args = parse_args()
brain = Brain(args.mode, args.circuit)
 
def user_main():
    brain.execute()


def main():
    
    HAL.setW(0.0)
    HAL.setV(0.0)
    HAL.main(user_main)


# Execute!
if __name__ == "__main__":
    main()
