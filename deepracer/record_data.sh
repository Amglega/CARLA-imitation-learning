 #!/bin/bash

source /opt/ros/foxy/setup.bash
source /home/deepracer/deepracer_nav2_ws/aws-deepracer/install/setup.bash
export ROS_DOMAIN_ID=5
printenv | grep -i ROS | egrep ROS_DOMAIN_ID

python3 deepracer_record_local_data.py
