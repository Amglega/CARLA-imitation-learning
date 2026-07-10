 #!/bin/bash

systemctl stop deepracer-core

source /opt/ros/foxy/setup.bash
source /home/deepracer/deepracer_nav2_ws/aws-deepracer/install/setup.bash
export ROS_DOMAIN_ID=5
printenv | grep -i ROS | egrep ROS_DOMAIN_ID

ros2 launch servo_pkg servo_pkg_launch.py
