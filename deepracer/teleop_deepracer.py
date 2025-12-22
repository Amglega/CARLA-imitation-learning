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
import utils.hal as HAL
import csv
import argparse
import threading

#forward_key_pressed, backward_key_pressed, left_key_pressed, right_key_pressed = False

settings = termios.tcgetattr(sys.stdin)

Finish_program = False

THROTLE_VAL = 0.55
linear_speed = THROTLE_VAL
angular_speed = 0

class GetKeyThread(threading.Thread):

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key


    def run(self):
        global linear_speed
        global angular_speed
        global Finish_program

        while (1):
            key = self.getKey()
            
            if key == "w":
                if linear_speed < 18:
                    linear_speed = THROTLE_VAL
                    if angular_speed > 0:
                        angular_speed = 1.0
                    else:
                        angular_speed = -1.0
            if key == "s":
                if linear_speed > -2 :
                    linear_speed = -(THROTLE_VAL-0.1)
                    if angular_speed > 0:
                        angular_speed = 1.0
                    else:
                        angular_speed = -1.0
            if key == "a":
                if angular_speed < 0:
                    angular_speed = 0.1
                elif angular_speed > 0 and angular_speed < 1:
                    angular_speed = angular_speed + 0.1
                elif angular_speed == 0:
                    angular_speed = angular_speed + 0.1
            elif key == "d":
                if angular_speed > 0:
                    angular_speed = - 0.1
                elif angular_speed < 0 and angular_speed > -1:
                    angular_speed = angular_speed - 0.1
                elif angular_speed == 0:
                    angular_speed = angular_speed - 0.1
            elif key == "q":
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
                Finish_program = True
                sys.exit()


class CameraThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.coordinates_d = (450,120)
        self.coordinates_w = (450,100)
        self.coordinates_v = (450,80)

    def check_center(self, position_x):
        if len(position_x[0]) > 1:
            x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
            not_found = False
        else:
            # The center of the line is in position 326
            x_middle = 326
            not_found = True
        return x_middle, not_found

    def get_point(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right) / 2 + left
        return int(mid)

    def run(self):
        global linear_speed
        global angular_speed
        global Finish_program
        while (1):
            if Finish_program:
                sys.exit(0)
            image = HAL.getImage()
            if image.shape[0] > 50:
           
                image_cropped = image[230:, :, :]

                image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0, 50, 50])
                upper_red = np.array([180, 255, 255])
                image_mask = cv2.inRange(image_hsv, lower_red, upper_red)

                rows, cols = image_mask.shape
                rows = rows - 1  # para evitar desbordamiento

                alt = 0
                ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
                if np.count_nonzero(ff[:, 0]) > 0:
                    alt = np.min(np.nonzero(ff[:, 0]))

                points = []
                for i in range(3):
                    if i == 0:
                        index = alt
                    else:
                        index = rows // (2 * i)
                    points.append((self.get_point(index, image_mask), index))

                points.append((self.get_point(rows, image_mask), rows))

                # We convert to show it
                # Shape gives us the number of rows and columns of an image
                size = image_mask.shape
                rows = size[0]
                columns = size[1]

                # We look for the position on the x axis of the pixels that have value 1 in different positions and
                position_x_down = np.where(image_mask[points[3][1], :])
                position_x_middle = np.where(image_mask[points[1][1], :])
                position_x_above = np.where(image_mask[points[2][1], :])

                # We see that white pixels have been located and we look if the center is located
                # This way we can know if the car has left the circuit
                x_middle_left_down, not_found_down = self.check_center(position_x_down)
                x_middle_left_middle, not_found_middle = self.check_center(position_x_middle)
        
                # We look if white pixels of the row above are located
                if (len(position_x_above[0]) > 1):
                    self.x_middle_left_above = (position_x_above[0][0] + position_x_above[0][
                        len(position_x_above[0]) - 1]) / 2
                    # We look at the deviation from the central position. The center of the line is in position cols/2
                    deviation = self.x_middle_left_above - (cols / 2)
                    
                    # We update the deviation
                    self.deviation_left = deviation
                    text_d = "Deviation:" + str(deviation)
                else:
                    text_d = "OUT OF THE TRACK"
                text_w = "Angular speed:" + str(angular_speed)
                text_v = "Linear speed:" + str(linear_speed)
                cv2.putText(image, text_w, self.coordinates_w, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 1, cv2.LINE_AA)
                cv2.putText(image, text_v, self.coordinates_v, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 1, cv2.LINE_AA)
                cv2.putText(image, text_d, self.coordinates_d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0) , 1, cv2.LINE_AA)
                cv2.imshow("Camara",image)
                cv2.waitKey(1)

            else:
                time.sleep(3)


class Brain:

    def __init__(self, mode=None, circuit = "simple"):
        threading.Thread.__init__(self)
        self.iteration = 0
        self.circuit = circuit 
        self.path = os.getcwd() + "/" + "datasets"
        self.mode = mode

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


    def execute(self):
        global linear_speed
        global angular_speed
        global Finish_program
        
        image = HAL.getImage()
            
        if image.shape[0] > 50:

            HAL.setV(linear_speed)
            HAL.setW(angular_speed)

            if self.mode == "save":
                self.iteration += 1
                cv2.imwrite(self.path + "/" + str(self.iteration) + ".png", image)
                self.writer_output.writerow([str(self.iteration) + '.png',linear_speed,angular_speed])
        else:
            time.sleep(1)
        

def end_signal_handler(sig, frame):
    global Finish_program
    Finish_program = True


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--mode", type=str, default=None, help="To save or not to save the data")
    parser.add_argument("--circuit", type=str, default="simple", help="Name of the circuit")

    args = parser.parse_args()
    return args

args = parse_args()
brain = Brain(args.mode, args.circuit)

def user_main():
    global Finish_program
    brain.execute()
    if Finish_program:
        HAL.setV(0)
        HAL.setW(0)
        cv2.destroyAllWindows() 
        sys.exit(0)



def main():
    
    key_thread = GetKeyThread()
    key_thread.start()
    cam_thread = CameraThread()
    cam_thread.start()
    signal.signal(signal.SIGINT, end_signal_handler)
    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)
    

# Execute!
if __name__ == "__main__":
    main()
