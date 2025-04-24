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
    
    def close(self):
        """
        Close the PyBullet simulation and clean up resources.
        """
        # Disconnect from the PyBullet server
        pybullet.disconnect()

        # Clear the objects dictionary
        self.objects.clear()

        # Clear the constraints dictionary
        self.constraints.clear()

        # Clear the collision matrix
        self.collision_matrix = None

        # Clear the box info
        self.box_info = None

        # Clear the step data
        self.step_data.clear()

        # Reset the current step
        self.current_step = 0

        # Clear the robots list
        self.robots.clear()

        # Clear the simulation data
        self.simulation_data.clear()
        self.normalized_simulation_data.clear()

        # Clear the reward list
        self.reward.clear()

        print("PyBullet simulation closed and resources cleaned up.")
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
        normalized_data = self.normalize_data(step_data)
        self.normalized_simulation_data.append(normalized_data)
        self.current_step += 1

        if self.normalized_simulation_data:
            latest_data = self.normalized_simulation_data[-1]
            return np.array(latest_data, dtype=np.float32).flatten()
        else:
            return np.zeros(455, dtype=np.float32).flatten()

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
    def setup(self,item_scale):
        # Define box size limits and spawn height
        min_box_size = Item.get_scaled_min_dimensions(item_scale)
        max_box_size = Item.get_scaled_max_dimensions(item_scale)
        #random_box_size = Item.get_random_dimensions(item_scale)
        #min_box_size = Item.MIN_DIMENSIONS*item_scale
        #max_box_size = Item.MAX_DIMENSIONS*item_scale
        distance_below_robots = Item.HEIGHT_FROM_ROBOT  # Distance from robot spawn height to top of box

        # Generate random box size
        box_size = [random.uniform(min_box_size[i], max_box_size[i]) for i in range(3)]
        
        # Calculate box position
        robot_height = RobotTeam.CENTER[2]  # Assuming this is the height where robots are spawned
        box_z_position = robot_height - distance_below_robots - box_size[2]
        box_position = [Item.X_Y_CENTER[0], Item.X_Y_CENTER[1], box_z_position]

        # Spawn the box
        box_id = self.spawn_box(box_position, box_size, Item.MASS, Item.COLOR, "main_box")
        objects_to_grasp = [box_id]

        # Create a constraint to hold the box in place
        self.create_constraint(box_id, box_position)
        initial_velocities = [([0, 0, 0], [0, 0, 0]) for _ in range(len(self.robots))]
        initial_suction_commands = [0 for _ in range(len(self.robots))]
        self.record_simulation_data(initial_velocities, initial_suction_commands,self.goal)
        robot_controller = RobotController(self.robots)

        # Calculate circular pattern radius (half the smallest horizontal dimension of the box)
        circle_radius = min(box_size[0], box_size[1]) / 4

        # Generate target points for circular movement above the box
        box_top_z = box_position[2] + box_size[2]/2
        target_points1 = generate_circular_points([0, 0, 0], circle_radius, RobotTeam.NUM_ROBOTS, box_top_z + 0.1)
        target_points2 = generate_circular_points([0, 0, 0], circle_radius, RobotTeam.NUM_ROBOTS, box_top_z)
        target_points3 = generate_circular_points([0, 0.2, 0], circle_radius, RobotTeam.NUM_ROBOTS, box_top_z)

        initial_orientations = [pybullet.getQuaternionFromEuler([0, math.pi, 0]) for _ in range(RobotTeam.NUM_ROBOTS)]
        robot_controller.move_robots_simultaneously(list(zip(target_points1, initial_orientations)))
        robot_controller.move_robots_simultaneously(list(zip(target_points2, initial_orientations)))

        print("Robots moved to initial positions")
        time.sleep(1)
        success_counter = 0
        for robot in self.robots:
            success = robot.activate_suction(objects_to_grasp[0])
            if success:
                success_counter += 1
    
        print(f"Suction Successes: {success_counter}")
        self.remove_constraint(box_id)
        robot_controller.move_robots_simultaneously(list(zip(target_points1, initial_orientations)))
        time.sleep(1)
        #robot_controller.move_robots_simultaneously(list(zip(target_points3, initial_orientations)))

        for robot in self.robots:
            robot.deactivate_suction()
        print("done suction")

        return robot_controller, box_id
    def is_done(self, state):
        box_orientation = self.get_box_pose()['orientation']
        target_orientation = self.goal
        box_position = self.get_box_pose()['position']

        # Check if the box has reached the target orientation
        if np.dot(box_orientation, target_orientation) > 0.99:
            return True
        
        # Check if the box has moved outside the workspace (using 3D distance)
        if np.linalg.norm(box_position - np.array(RobotTeam.CENTER)) > SimulationConfig.WORKSPACE_RADIUS:
            return True
        
        # Check if maximum steps have been reached
        if self.current_step >= SimulationConfig.NUM_STEPS:
            return True
        
        return False
    def normalize_data(self, step_data):
        normalized_data = []

        # Normalize time (1 element)
        normalized_data.append(Normalizer.normalize_value(step_data['time'], SimulationConfig.NUM_STEPS * SimulationConfig.DT))
        
        # Normalize robot states (4 robots * (6 joint angles + 6 joint velocities + 3 position + 4 orientation + 3 velocity + 3 angular velocity + 3 suction states) = 112 elements)
        for robot_data in step_data['robot_states']:
            robot_state = robot_data['robot_state']
            normalized_data.extend([Normalizer.normalize_value(angle, 2*np.pi) for angle in robot_state['joint_angles']])
            normalized_data.extend([Normalizer.normalize_value(vel, RobotConfig.MAX_VELOCITY) for vel in robot_state['joint_velocities']])
            normalized_data.extend([Normalizer.normalize_value(pos, SimulationConfig.WORKSPACE_RADIUS) for pos in robot_state['position']])
            normalized_data.extend(robot_state['orientation'])
            normalized_data.extend([Normalizer.normalize_value(vel, RobotConfig.MAX_EE_VELOCITY) for vel in robot_state['velocity']])
            normalized_data.extend([Normalizer.normalize_value(avel, RobotConfig.MAX_EE_AVELOCITY) for avel in robot_state['angular_velocity']])
            normalized_data.extend([robot_state['suction_on'], robot_state['suction_accident'], robot_state['suction_failure']])


        # Normalize collision matrix (upper triangle, excluding diagonal) 300 elements, 25 objects
        if step_data['collision_matrix'] is not None:
            n = len(step_data['collision_matrix'])
            for i in range(n):
                for j in range(i + 1, n):
                    normalized_data.append(step_data['collision_matrix'][i][j])



        # Normalize box info (10 elements)
        if step_data['box_info'] is not None:
            normalized_data.extend([Normalizer.normalize_value(pos, SimulationConfig.WORKSPACE_RADIUS) for pos in step_data['box_info']['position']])
            normalized_data.extend(step_data['box_info']['orientation'])
            normalized_data.extend([Normalizer.normalize_value(size, max(Item.MAX_DIMENSIONS)) for size in step_data['box_info']['size']])
        else:
            normalized_data.extend([0] * 10)  # If box_info is None, add zeros


        # Normalize goal (4 elements)
        normalized_data.extend(step_data['goal'])

        # Normalize velocities (4 robots * (3 linear + 3 angular) = 24 elements)
        for vel in step_data['velocities']:
            normalized_data.extend([Normalizer.normalize_value(v, RobotConfig.MAX_EE_VELOCITY) for v in vel[0]])
            normalized_data.extend([Normalizer.normalize_value(v, RobotConfig.MAX_EE_AVELOCITY) for v in vel[1]])

        # Normalize suction commands (4 elements)
        normalized_data.extend(step_data['suction_commands'])


        # Ensure we have exactly 455 elements
        assert len(normalized_data) == 455, f"Expected 455 elements, but got {len(normalized_data)}"
        
        return normalized_data