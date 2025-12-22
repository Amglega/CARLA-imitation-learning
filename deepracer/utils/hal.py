
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from deepracer_interfaces_pkg.msg import ServoCtrlMsg
from deepracer_interfaces_pkg.srv import SetMaxSpeedSrv
from sensor_msgs.msg import Image
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import cv2
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import math
from std_msgs.msg import String
import time
from datetime import datetime
from utils.constants import *


FREQUENCY = 60

class DeepRacerNode(Node):

    def __init__(self, frecuency):
        super().__init__("velocity_publisher_node")

        self.target_linear = 0.0
        self.target_rot = 0.0
        self.action_pub = self.create_publisher(ServoCtrlMsg, ACTION_PUBLISH_TOPIC, 10)    
        self.timer = self.create_timer(1/frecuency, self.timer_callback)

        #camera atributes:
        self.vid = cv2.VideoCapture(0)
        self.image = np.zeros((3, 3, 3), np.uint8)
        self.img_shape = (640, 480)
        self.K = np.array([[502.4827132565883, 0.0, 320.49002418357725], [0.0, 502.4546524395416, 238.255941996664], [0.0, 0.0, 1.0]])
        self.D = np.array([[-0.08703736838521056], [-0.2917938213212864], [0.6776229437062419], [-0.3476415479534463]])
        
    # -- Transform Vel Methods --
    #####################
     
    def plan_action(self):
        """Calculate the target steering and throttle.

        Returns:
            steering (float): Angle value to be published to servo.
            throttle (float): Throttle value to be published to servo.
        """

        throttle = self.target_linear
        if self.target_linear < 0:
            throttle = -0.4
        elif self.target_linear > ActionValues().MAX_THROTTLE_OUTPUT :
            throttle = ActionValues().MAX_THROTTLE_OUTPUT * math.copysign(1.0, self.target_linear)
        elif self.target_linear < ActionValues().MIN_THROTTLE_OUTPUT:
            throttle = 0.0

   
       
        # Set the direction.
        steering = self.target_rot
        
        if abs(self.target_rot) > ActionValues().MAX_STEERING_OUTPUT :
            steering = ActionValues().MAX_STEERING_OUTPUT * math.copysign(1.0, self.target_rot)

            
        if self.target_linear < 0:
            steering = steering *-1

        return steering, throttle
    #####################
    
    
    def getV(self):
        return self.target_linear
    
    def getW(self):
        return self.target_rot

    def set_V(self, num):
        """
        Args:
            num: linear speed
        """
        self.target_linear = num
        
    def set_W(self, num):
        """
        Args:
            num: angular speed
        """
        self.target_rot = num
    
    def action_publish(self, target_steer, target_speed):
        """Function publishes the action and sends it to servo.

        Args:
            target_steer (float): Angle value to be published to servo.
            target_speed (float): Throttle value to be published to servo.
        """
        result = ServoCtrlMsg()
        result.angle, result.throttle = float(target_steer),float(target_speed)
        
        #self.get_logger().info(f"Publishing to servo: Steering {target_steer} | Throttle {target_speed}")
        self.action_pub.publish(result)

    def timer_callback(self):
        try:
            target_steer, target_speed = self.plan_action()
            self.action_publish(target_steer, target_speed)
        except Exception as ex:
            self.get_logger().error(f"Failed to publish action: {ex}")
            self.action_publish(ActionValues.DEFAULT_OUTPUT, ActionValues.DEFAULT_OUTPUT)
        
        ret, frame = self.vid.read() 
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.img_shape, cv2.CV_16SC2)
        self.image = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# -- Camera Method --
#####################

    def getImage(self):
        return self.image

#####################
 



# -- HAL Functions --
#####################

def getW():
    return deepracer_node.getW()

def getV():
    return deepracer_node.getV()

def setV(num):
    deepracer_node.set_V(num)
    
def setW(num):
    deepracer_node.set_W(num)

def getImage():
    return deepracer_node.getImage()

    
#init ros
rclpy.init()

# Create the nodes
deepracer_node = DeepRacerNode(FREQUENCY)

# --MAIN--
#####################

def main(user_main=None,n_threads=1,args=None):
    #Create executor
    executor = MultiThreadedExecutor(num_threads=n_threads)
    executor.add_node(deepracer_node)

    time_cycle = 1000.0 / FREQUENCY

    try:
        while rclpy.ok():

            start_time = datetime.now()
            
            user_main()
            executor.spin_once()
            
            finish_time = datetime.now()
            dt = finish_time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0

            if(ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)          
    except KeyboardInterrupt:
        pass
    finally:
        #stop the motors before exit
        setV(0.0)
        setW(0.0)
        executor.spin_once()
        time.sleep(0.1)
        executor.shutdown()
        deepracer_node.destroy_node()
        # Shutdown the ROS client library for Python
        rclpy.shutdown()
        exit()
        

    
if __name__ == '__main__':
    main()
