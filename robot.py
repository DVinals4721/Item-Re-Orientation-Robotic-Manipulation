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


class UR5Robot(Robot):
    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        return pybullet.loadURDF(RobotConfig.URDF_PATH, self.position, self.orientation, flags=flags)

    def get_end_effector_index(self):
        for i in range(self.num_joints):
            if pybullet.getJointInfo(self.robot_id, i)[12].decode('utf-8') == "suction_tip":
                return i
        raise ValueError("Suction cylinder link not found in the URDF file")