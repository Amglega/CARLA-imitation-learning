# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from rclpy.executors import MultiThreadedExecutor
import geometry_msgs.msg
import numpy as np
from std_msgs.msg import String
import time
from datetime import datetime

class VelocityPublisher(Node):

    def __init__(self):
        super().__init__("velocity_publisher_node")

        self.v = 0.0
        self.w = 0.0
        self.pub = self.create_publisher(geometry_msgs.msg.Twist, '/f1ros2/cmd_vel', 10)

        
        self.timer = self.create_timer(0.05, self.timer_callback)
    
    def getV(self):
        return self.v
    
    def getW(self):
        return self.w
    
    def setV(self, num):
        self.v = num

    def setW(self, num):
        self.w = num
    
    def timer_callback(self):
        msg = geometry_msgs.msg.Twist()
        msg.linear.x = float(self.v)
        msg.angular.z = float(self.w)
        self.pub.publish(msg)


class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """
    def __init__(self):
    # Initiate the Node class's constructor and give it a name
        super().__init__('image_subscriber')
        
        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
        Image, 
        '/f1ros2/cam_f1_left/image_raw', 
        self.listener_callback, 
        10)
        self.subscription # prevent unused variable warning
        self.image = np.zeros((3, 3, 3), np.uint8)
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def listener_callback(self, data):
        # Display the message on the console
        #self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        self.image = self.br.imgmsg_to_cv2(data,"bgr8")

    def getImage(self):
        return self.image

rclpy.init()

# Create the nodes
image_subscriber = ImageSubscriber()
velocity_publisher = VelocityPublisher()

# -- HAL Functions --
#####################

def getImage():
    return image_subscriber.getImage()

def getW():
    return velocity_publisher.getW()

def getV():
    return velocity_publisher.getV()

def setV(num):
    velocity_publisher.setV(num)

def setW(num):
    velocity_publisher.setW(num)

# --MAIN--
#####################

def main(user_main,n_threads=1,args=None):
    #Create executor
    executor = MultiThreadedExecutor(num_threads=n_threads)
    executor.add_node(image_subscriber)
    executor.add_node(velocity_publisher)
    frequency = 60
    time_cycle = 1000.0 / frequency
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
        executor.shutdown()
        image_subscriber.destroy_node()
        velocity_publisher.destroy_node()
        # Shutdown the ROS client library for Python
        rclpy.shutdown()
        

    
if __name__ == '__main__':
    main()