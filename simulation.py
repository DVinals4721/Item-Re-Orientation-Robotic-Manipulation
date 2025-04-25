from config import SimulationConfig, RobotTeam, Item, RobotConfig
import pybullet
import pybullet_data
import numpy as np
import math
from robot import UR5Robot
import random
import time
from robot_controller import RobotController
from utils import generate_circular_points, Normalizer

class SimulationEnvironment:
    def __init__(self, box_quaternion_angles, use_gui=False):
        if use_gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(*SimulationConfig.GRAVITY)
        #self.plane_id = pybullet.loadURDF("plane.urdf")
        #self.objects = {"plane": self.plane_id}
        self.objects = {}
        self.collision_matrix = None
        self.box_info = None
        self.step_data = []
        self.current_step = 0
        self.robots = self.spawn_robots_on_circle()
        self.constraints = {}
        self.simulation_data = []
        self.normalized_simulation_data = []
        self.reward = []
        self.goal = box_quaternion_angles
    

    def spawn_box(self, position, size, mass, color, name, enable_collision=False):
        """
        Spawn a box with optional collision detection.
        
        Parameters:
            position: [x, y, z] position of the box
            size: [length, width, height] dimensions
            mass: Mass of the box
            color: RGBA color of the box
            name: Identifier for the box
            enable_collision: If False, the box will be visible but have no physical collision
        """
        visual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=[s/2 for s in size],
            rgbaColor=color
        )
        
        if enable_collision:
            collision_shape_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=[s/2 for s in size]
            )
        else:
            collision_shape_id = -1  # No collision shape
        
        box_id = pybullet.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        self.objects[name] = box_id
        self.box_info = {
            'id': box_id,
            'name': name,
            'size': size
        }
        return box_id
    def get_box_pose(self):
        if self.box_info:
            pos, orn = pybullet.getBasePositionAndOrientation(self.box_info['id'])
            return {
                'position': pos,
                'orientation': orn,
                'size': self.box_info['size']
            }
        return None

    def check_collisions(self):
        def get_contacts(obj1, obj2):
            if isinstance(obj1, tuple) and isinstance(obj2, tuple):
                return pybullet.getContactPoints(obj1[0], obj2[0], obj1[1], obj2[1]) if obj1[0] != obj2[0] else []
            elif isinstance(obj1, tuple):
                return pybullet.getContactPoints(obj1[0], obj2, linkIndexA=obj1[1])
            elif isinstance(obj2, tuple):
                return pybullet.getContactPoints(obj1, obj2[0], linkIndexB=obj2[1])
            else:
                return pybullet.getContactPoints(obj1, obj2)
        n = len(self.objects)
        #print(f"object number {self.objects}")
        #print(f"objects {n}")
        self.collision_matrix = np.zeros((n, n), dtype=int)
        object_keys = list(self.objects.keys())

        for i in range(n):
            for j in range(i + 1, n):
                obj1 = self.objects[object_keys[i]]
                obj2 = self.objects[object_keys[j]]
                contacts = get_contacts(obj1, obj2)
                if contacts:
                    self.collision_matrix[i, j] = self.collision_matrix[j, i] = 1
        #print(f"Collision matrix Size {self.collision_matrix.shape}")

    def spawn_robots_on_circle(self):
        center = RobotTeam.CENTER
        radius = RobotTeam.ROBOT_RADIUS
        num_robots = RobotTeam.NUM_ROBOTS
        num_robots = 4
        robots = []
        for i in range(num_robots):
            if i == 0:
                angle = 2*math.pi * i / num_robots 
                position = [center[0] + radius * math.cos(angle),
                            center[1] + radius * math.sin(angle),
                            center[2]]
                orientation = pybullet.getQuaternionFromEuler([0, 0, angle-math.pi ])
            if i == 1:
                    angle = 2*math.pi * i / num_robots 
                    position = [center[0] + radius * math.cos(angle),
                                center[1] + radius * math.sin(angle),
                                center[2]]
                    orientation = pybullet.getQuaternionFromEuler([0, 0, angle + math.pi])
            if i == 2:
                angle = 2*math.pi * i / num_robots 
                position = [center[0] + radius * math.cos(angle),
                            center[1] + radius * math.sin(angle),
                            center[2]]
                orientation = pybullet.getQuaternionFromEuler([0, 0, angle])
            if i == 3:
                    angle = 2*math.pi * i / num_robots 
                    position = [center[0] + radius * math.cos(angle),
                                center[1] + radius * math.sin(angle),
                                center[2]]
                    orientation = pybullet.getQuaternionFromEuler([0, 0, angle+math.pi ])

            
            robot= UR5Robot(position, orientation)
            robots.append(robot)
            for joint_index in robot.joint_indices:
                link_name = f"robot_{i}_Link_{joint_index}"
                self.objects[link_name] = (robot.robot_id, i)
        return robots

    def record_simulation_data(self, velocities, suction_commands, box_quaternion_angles):
        self.check_collisions()
        robot_states = []
        for robot in self.robots:
            robot_state = {
                'robot_state': robot.get_robot_states()
            }
            robot_states.append(robot_state)

        # Get current box pose
        current_box_pose = self.get_box_pose()
        
        step_data = {
            'time': self.current_step * SimulationConfig.DT,
            'robot_states': robot_states,
            'collision_matrix': self.collision_matrix.tolist() if self.collision_matrix is not None else None,
            'box_info': current_box_pose,
            'goal':box_quaternion_angles,
            'velocities': velocities,
            'suction_commands': suction_commands
        }
        self.simulation_data.append(step_data)



    def create_constraint(self, object_id, position):
        constraint_id = pybullet.createConstraint(
            object_id, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position
        )
        self.constraints[object_id] = constraint_id
        return constraint_id

    def remove_constraint(self, object_id):
        if object_id in self.constraints:
            pybullet.removeConstraint(self.constraints[object_id])
            del self.constraints[object_id]
    
