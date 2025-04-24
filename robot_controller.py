import threading
from utils import generate_circular_points
import pybullet
import math
from robot import UR5Robot
import numpy as np

class RobotController:
    def __init__(self, robots):
        self.robots = robots

    def move_robots_simultaneously(self, new_poses,vel=0.5):
        threads = [threading.Thread(target=robot.move_to_pose, args=(pose[0], pose[1], vel))
                   for robot, pose in zip(self.robots, new_poses)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def control_robots_simultaneously(self, velocities, suction_commands, object):
     
        threads = []
        for robot, vel, suction in zip(self.robots, velocities, suction_commands):
            thread = threading.Thread(target=self.apply_commands, args=(robot, vel[0], vel[1],suction,object))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return velocities, suction_commands

    def apply_commands(self, robot, linear_velocity, angular_velocity, suction_command,object):
        print(f"lv: {linear_velocity}")
        print(f"la: {angular_velocity}")
        print(f"sc: {suction_command}")
        robot.move_with_velocity_step(linear_velocity, angular_velocity)  # Assuming no angular velocity
        if suction_command == 1:
            robot.activate_suction(object)
        elif suction_command == 0:
            robot.deactivate_suction()


