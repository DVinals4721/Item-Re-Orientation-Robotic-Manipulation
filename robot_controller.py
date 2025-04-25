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




