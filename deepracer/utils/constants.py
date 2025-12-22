#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################

ACTION_PUBLISH_TOPIC = "/ctrl_pkg/servo_msg"
SET_MAX_SPEED_SERVICE_NAME = "set_max_speed"

CMDVEL_TOPIC = "/cmd_vel"



class ActionValues():
    """Class with the PWM values with respect to
       the possible actions that can be sent to servo, pertaining to
       the angle and throttle.
    """
    MIN_THROTTLE_OUTPUT = 0.5
    MAX_THROTTLE_OUTPUT = 0.8
    MAX_STEERING_OUTPUT = 1.0
    DEFAULT_OUTPUT = 0.0

