from abc import ABC, abstractmethod
from config import RobotConfig, SimulationConfig
import pybullet
import numpy as np
from utils import visualize_frame
import time
from scipy.spatial.transform import Rotation

class Robot(ABC):
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation
        self.robot_id = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.robot_id)
        self.end_effector_index = self.get_end_effector_index()
        self.joint_indices = self.get_joint_indices()

        # Suction cup properties
        self.suction_diameter = RobotConfig.SUCTION_DIAMETER
        self.suction_force = RobotConfig.MAX_FORCE
        self.suction_normal_threshold = RobotConfig.SUCTION_NORMAL_THRESH
        self.suction_distance_min = RobotConfig.SUCTION_D_MIN
        self.suction_distance_max = RobotConfig.SUCTION_D_MAX
        self.suction_surface_threshhold = RobotConfig.SUCTION_SURFACE_THRESH
        self.attached_object_id = None
        self.attachment_point = None
        self.suction_constraint = None
        
        self.suction_active = False
        self.suction_desuction_accident = False
        self.suction_failure = False

    @abstractmethod
    def load_robot(self):
        pass

    @abstractmethod
    def get_end_effector_index(self):
        pass
            

    def find_suction_target(self, object_id):
        ee_pos, ee_orn = self.get_end_effector_pose()
        ee_rot = Rotation.from_quat(ee_orn)
        ee_z_axis = ee_rot.apply([0, 0, 1])

        #print(f"End effector position: {ee_pos}")
        #print(f"End effector z-axis: {ee_z_axis}")

        aabb_min, aabb_max = pybullet.getAABB(object_id)
        
        #print(f"Checking object {object_id}")
        #print(f"AABB min: {aabb_min}, max: {aabb_max}")

        # Find the closest points between the end effector and the object
        closest_points = pybullet.getClosestPoints(self.robot_id, object_id, self.suction_distance_max)

        if not closest_points:
            print(f"No close points found for object {object_id}")
            return None, None, None

        # Get the closest point
        closest_point = min(closest_points, key=lambda x: x[8])  # x[8] is the distance

        hit_pos = closest_point[5]  # Position on B
        hit_normal = closest_point[7]  # Normal on B
        distance = closest_point[8]  # Distance between A and B

        print(f"Hit position: {hit_pos}")
        #print(f"Hit normal: {hit_normal}")
        #print(f"Distance to surface: {distance}")

        # Check if the surface normal and end effector z-axis are pointing towards each other
        normal_alignment = np.dot(hit_normal, -ee_z_axis)

        #print(f"Normal alignment: {normal_alignment}")

        if self.suction_distance_min <= distance <= self.suction_distance_max:
            pass
            print(f"Distance {distance} within range [{self.suction_distance_min}, {self.suction_distance_max}]")
        else:
            print(f"Distance {distance} not within range [{self.suction_distance_min}, {self.suction_distance_max}]")
            return None, None, None

        if normal_alignment >= self.suction_normal_threshold:
            pass
            #print(f"Normal alignment {normal_alignment} above threshold {self.suction_normal_threshold}")
        else:
            #print(f"Normal alignment {normal_alignment} below threshold {self.suction_normal_threshold}")
            return None, None, None

        # Check surface area
        suction_radius = self.suction_diameter / 2
        num_samples = 8  # Number of points to check around the circumference
        valid_points = 0

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            offset = [suction_radius * np.cos(angle), suction_radius * np.sin(angle), 0]
            
            # Rotate offset by the hit normal
            rot_matrix = Rotation.from_rotvec(np.cross([0, 0, 1], hit_normal)).as_matrix()
            rotated_offset = np.dot(rot_matrix, offset)
            
            sample_point = [hit_pos[j] + rotated_offset[j] for j in range(3)]
            
            # Cast a ray from slightly above the sample point towards the object
            ray_from = [sample_point[j] + hit_normal[j] * 0.01 for j in range(3)]  # 1cm above surface
            ray_to = [sample_point[j] - hit_normal[j] * 0.01 for j in range(3)]  # 1cm below surface
            
            ray_test = pybullet.rayTest(ray_from, ray_to)[0]
            
            if ray_test[0] == object_id:
                valid_points += 1

        surface_coverage = valid_points / num_samples
        #print(f"Surface coverage: {surface_coverage:.2f}")

        if surface_coverage >= self.suction_surface_threshhold :  # At least 75% of points should be on the surface
            print(f"Suitable suction target found on object {object_id}")
            return object_id, hit_pos, hit_normal
        else:
            print(f"Insufficient surface area for suction on object {object_id}")
            return None, None, None

    def activate_suction(self, object_id):
        # Don't re-activate if already active with the same object
        if self.suction_active and self.attached_object_id == object_id:
            return True
            
        # Reset state for new activation
        self.attached_object_id = None
        self.attachment_point = None
        self.suction_constraint = None
        self.suction_failure = False
        
        print(f"Attempting suction connection with object {object_id}")
        object_id, point, normal = self.find_suction_target(object_id)
        
        if object_id:
            self.suction_active = True
            self.attached_object_id = object_id
            self.attachment_point = point

            # Create a constraint to simulate suction
            ee_pos, ee_orn = self.get_end_effector_pose()
            inv_ee_pos, inv_ee_orn = pybullet.invertTransform(ee_pos, ee_orn)
            
            # Calculate the relative position and orientation of the attachment point
            rel_pos, rel_orn = pybullet.multiplyTransforms(inv_ee_pos, inv_ee_orn, point, [0, 0, 0, 1])
            print(f"Creating suction constraint at distance: {np.linalg.norm(np.array(point) - np.array(ee_pos)):.4f}m")
            
            self.suction_constraint = pybullet.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.end_effector_index,
                childBodyUniqueId=object_id,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=rel_pos,  # Position relative to end effector
                childFramePosition=[0, 0, 0],  # Use the point on the object
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=[0, 0, 0, 1]
            )
            pybullet.changeConstraint(self.suction_constraint, maxForce=self.suction_force)
            self.suction_desuction_accident = False
            return True
        else:
            self.suction_failure = True
            return False

    def deactivate_suction(self):
        if self.suction_active:
            self.suction_active = False
            self.suction_desuction_accident = False
            self.suction_failure = False
            if self.suction_constraint is not None:
                pybullet.removeConstraint(self.suction_constraint)
                self.suction_constraint = None
            self.attached_object_id = None
            self.attachment_point = None
            print("Suction deactivated")

    def update_suction(self):
        if self.suction_active:
            if self.attached_object_id is not None and self.attachment_point is not None and self.suction_constraint is not None:
                ee_pos, ee_orn = self.get_end_effector_pose()
                obj_pos, obj_orn = pybullet.getBasePositionAndOrientation(self.attached_object_id)
                
                # Check if the object is still within suction range
                distance = np.linalg.norm(np.array(ee_pos) - np.array(self.attachment_point))
                if distance > self.suction_distance_max:
                    #print(f"Object detached: distance {distance:.4f} m exceeds maximum suction distance")
                    self.suction_desuction_accident = True
                
                    #self.deactivate_suction()
                else:
                    # Update constraint to maintain suction
                    #pybullet.changeConstraint(self.suction_constraint, ee_pos, ee_orn, maxForce=self.suction_force)
                    self.suction_desuction_accident = False

    def get_end_effector_pose(self):
        return pybullet.getLinkState(self.robot_id, self.end_effector_index)[:2]

    def get_joint_indices(self):
        return [pybullet.getJointInfo(self.robot_id, i)[0] for i in range(self.num_joints)
                if pybullet.getJointInfo(self.robot_id, i)[1].decode('utf-8') in RobotConfig.CONTROL_JOINTS]

    def set_joint_angles(self, joint_angles, velocities=None):
        if velocities is None:
            velocities = [0] * len(joint_angles)
        for i, (angle, velocity) in enumerate(zip(joint_angles, velocities)):
            pybullet.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=self.joint_indices[i],
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=angle,
                targetVelocity=velocity,
                force=RobotConfig.MAX_FORCE,
                maxVelocity=RobotConfig.MAX_VELOCITY,
                positionGain=1,
                velocityGain=1
            )
    def move_to_pose(self, position, orientation, velocity=3.0):
        target_joint_angles = self.calculate_ik(position, orientation)
        #print("IK solution:", target_joint_angles)
        joint_states = self.get_joint_states()
        current_joint_angles = [state[0] for state in joint_states]
        #print("Current joint angles:", current_joint_angles)
        
        max_diff = max([abs(t - c) for t, c in zip(target_joint_angles, current_joint_angles)])
        num_steps = int(max_diff / (RobotConfig.MAX_VELOCITY * SimulationConfig.DT)) + 1
        
        for step in range(num_steps):
            t = (step + 1) / num_steps
            interpolated_angles = [current_joint_angles[i] + t * (target_joint_angles[i] - current_joint_angles[i]) 
                                for i in range(len(current_joint_angles))]
            
            self.set_joint_angles(interpolated_angles)
            #self.update_suction()
            pybullet.stepSimulation()
            time.sleep(SimulationConfig.DT)


    def get_joint_states(self):
        return [(state[0], state[1]) for state in pybullet.getJointStates(self.robot_id, self.joint_indices)]
    def get_robot_states(self):
        joint_angles = []
        joint_velocities = []
        end_effector_state = None

        for joint_index in self.joint_indices:  # Use joint_indices instead of range(self.num_joints)
            # Get joint state
            state = pybullet.getJointState(self.robot_id, joint_index)
            joint_angles.append(state[0])  # Joint position (angle or displacement)
            joint_velocities.append(state[1])  # Joint velocity

        # Get end effector state
        link_state = pybullet.getLinkState(self.robot_id, self.end_effector_index, computeLinkVelocity=1)
        
        # Extract relevant information for the end effector
        end_effector_state = {
            'position': link_state[0],  # Position relative to base frame
            'orientation': link_state[1],  # Orientation relative to base frame
            'velocity': link_state[6],  # Linear velocity
            'angular_velocity': link_state[7],  # Angular velocity
        }

        robot_state = {
            'joint_angles': joint_angles,
            'joint_velocities': joint_velocities,
            'position': end_effector_state['position'],
            'orientation': end_effector_state['orientation'],
            'velocity': end_effector_state['velocity'],
            'angular_velocity': end_effector_state['angular_velocity'],
            'suction_on': int(self.suction_active),
            'suction_accident': int(self.suction_desuction_accident),
            'suction_failure': int(self.suction_failure)
        }
        #print(f"robot state{robot_state}")
        return robot_state

    def calculate_ik(self, position, orientation):
        # Safety check for position
        if position is None or (isinstance(position, (list, tuple, np.ndarray)) and len(position) == 0):
            print(f"ERROR: Invalid position: {position}")
            return []  # Return empty list as fallback
        
        # Safety check for orientation
        if orientation is None or (isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 0):
            print(f"ERROR: Invalid orientation: {orientation}")
            return []  # Return empty list as fallback
            
        # Now it's safe to check position[0]
        if isinstance(position, (list, tuple, np.ndarray)) and len(position) > 0 and isinstance(position[0], (list, tuple, np.ndarray)):
            position = position[0]
        
        # Handle Euler angles vs quaternions
        if isinstance(orientation, (list, tuple, np.ndarray)):
            if len(orientation) == 3:  # Euler angles
                orientation = pybullet.getQuaternionFromEuler(orientation)
            elif len(orientation) > 0 and isinstance(orientation[0], (list, tuple, np.ndarray)):
                orientation = orientation[0]
                if len(orientation) == 3:  # Nested Euler angles
                    orientation = pybullet.getQuaternionFromEuler(orientation)
        
        try:
            return pybullet.calculateInverseKinematics(
                self.robot_id, self.end_effector_index, position, orientation,
                maxNumIterations=100, residualThreshold=1e-5
            )
        except pybullet.error as e:
            print(f"PyBullet IK error: {e}")
            print(f"Position: {position}, Orientation: {orientation}")
            return []  # Return empty list as fallback
    def visualize_frames(self):
        ee_state = pybullet.getLinkState(self.robot_id, self.end_effector_index)
        visualize_frame(ee_state[0], ee_state[1])
        base_pos, base_orn = pybullet.getBasePositionAndOrientation(self.robot_id)
        visualize_frame(base_pos, base_orn, line_length=0.2, line_width=3)

    def move_with_velocity_step(self, linear_velocity, angular_velocity_euler):
        # Get current end effector pose
        current_pos, current_orn = self.get_end_effector_pose()

        # Calculate delta position based on linear velocity and DT
        delta_position = [v * SimulationConfig.DT for v in linear_velocity]

        # Calculate delta orientation based on angular velocity and DT
        delta_orientation_euler = [w * SimulationConfig.DT for w in angular_velocity_euler]

        # Calculate new position
        new_position = [current_pos[i] + delta_position[i] for i in range(3)]

        # Calculate new orientation
        current_rotation = Rotation.from_quat(current_orn)
        delta_rotation = Rotation.from_euler('xyz', delta_orientation_euler)
        new_rotation = delta_rotation * current_rotation
        new_orientation = new_rotation.as_quat()

        # Calculate IK for the new pose
        target_joint_angles = self.calculate_ik(new_position, new_orientation)

        # Get current joint angles
        joint_states = self.get_joint_states()
        current_joint_angles = [state[0] for state in joint_states]

        # Set new joint angles
        self.set_joint_angles(target_joint_angles)
        self.update_suction()

        # Return the actual new pose achieved
        return self.get_end_effector_pose()
    
class UR5Robot(Robot):
    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        return pybullet.loadURDF(RobotConfig.URDF_PATH, self.position, self.orientation, flags=flags)

    def get_end_effector_index(self):
        for i in range(self.num_joints):
            if pybullet.getJointInfo(self.robot_id, i)[12].decode('utf-8') == "suction_tip":
                return i
        raise ValueError("Suction cylinder link not found in the URDF file")