# showcase.py
from config import SimulationConfig, RobotTeam, Item, RobotConfig
import pybullet
import pybullet_data
import numpy as np
import math
from robot import UR5Robot
import random
import time
import cv2
from robot_controller import RobotController
from utils import generate_circular_points, Normalizer
from simulation import SimulationEnvironment
from robot_controller import RobotController
import utils
from utils import generate_circular_points, Normalizer
from gait_generator import GaitGenerator
from scipy.spatial.transform import Rotation 
import threading

resolution = 0.05
clearance_radius = 0.07
num_arc_points = 8
min_intersection_distance = 0.22
arc_height_factor=2
box_size = np.array([1,1,1])
goal_orientation = np.array([np.pi/2,0, 0])
vel = 3

class VideoRecorder:
    def __init__(self, filename="simulation.mp4", fps=30, width=1280, height=720):
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        self.recording = True
        print(f"Recording started. Output: {filename}")
        
    def capture_frame(self, box_id=None):
        if not self.recording:
            return
            
        # Set camera parameters
        camera_distance = 2.0
        camera_yaw = 50
        camera_pitch = -35
        
        # Set camera target to box if available
        if box_id is not None:
            camera_target, _ = pybullet.getBasePositionAndOrientation(box_id)
        else:
            camera_target = [0, 0, 0]
            
        # Compute view and projection matrices
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.width)/self.height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture frame
        (_, _, px, _, _) = pybullet.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to RGB and write to video
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.height, self.width, 4))
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        self.writer.write(rgb_array)
        
    def stop(self):
        if self.recording:
            self.writer.release()
            self.recording = False
            print("Recording stopped and saved.")

def get_robot_commands(box_size, box_position, box_orientation, initial_robot_pos):
    generator = GaitGenerator(box_size, resolution, clearance_radius, num_arc_points,min_intersection_distance,arc_height_factor)
    # Generate paths
    generator.generate_paths(
        initial_centroid=np.array(box_position),
        initial_orientation=box_orientation,
        target_orientation=goal_orientation
    )

    # Get robot commands
    robot_commands = generator.assign_robots(initial_robot_pos)
    generator.plot_trajectory()
    # Return additional transformation data for box movement
    return {
        'commands': robot_commands,
        'shifts': generator.shifts,
        'rotation_vectors': generator.rotation_vectors,
        'initial_point': generator.initial_point
    }

def move_box_thread(box_id, position, orientation, smooth=True, steps=100):
    """Function to move a box in a separate thread, ensuring continuous motion"""
    # Convert orientation to quaternion if it's Euler angles
    if isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 3:
        orientation = pybullet.getQuaternionFromEuler(orientation)
    
    # Get current position and orientation - do this BEFORE removing any constraint
    current_pos, current_orn = pybullet.getBasePositionAndOrientation(box_id)
    
    print(f"Moving box from {current_pos} to {position}")
    
    if not smooth:
        # Even for instant movement, set directly from current position to target
        pybullet.resetBasePositionAndOrientation(box_id, position, orientation)
        pybullet.stepSimulation()
    else:
        # Smoothly interpolate between current and target pose
        for i in range(steps + 1):
            t = i / steps  # Interpolation parameter [0, 1]
            
            # Linear interpolation for position
            interp_pos = [
                current_pos[0] + t * (position[0] - current_pos[0]),
                current_pos[1] + t * (position[1] - current_pos[1]),
                current_pos[2] + t * (position[2] - current_pos[2])
            ]
            
            # Spherical linear interpolation for orientation
            interp_orn = pybullet.getQuaternionSlerp(current_orn, orientation, t)
            
            # Apply the interpolated pose
            pybullet.resetBasePositionAndOrientation(box_id, interp_pos, interp_orn)
            pybullet.stepSimulation()

def main():
    # Initialize simulation and video recorder
    sim_env = SimulationEnvironment(goal_orientation, use_gui=True)
    recorder = VideoRecorder("robot_manipulation.mp4")
    
    robots = sim_env.robots
    robot_controller = RobotController(sim_env.robots)

    # Generate box
    box_position = np.array([Item.X_Y_CENTER[0], Item.X_Y_CENTER[1], 
                           RobotTeam.CENTER[2] - Item.HEIGHT_FROM_ROBOT - box_size[2]/2])
    box_id = sim_env.spawn_box(box_position, box_size, Item.MASS, Item.COLOR, "main_box")
    
    # Initial recording
    recorder.capture_frame(box_id)
    
    # Create initial constraint to hold box in place
    box_constraint = pybullet.createConstraint(
        parentBodyUniqueId=box_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=pybullet.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=box_position
    )
    
    # Position robots around the box
    initial_robot_pos = generate_circular_points(
        box_position, radius=min(box_size[0], box_size[1]) / 4, 
        num_points=RobotTeam.NUM_ROBOTS, height=box_position[2] + box_size[2]/2 + 0.01
    )
    initial_orientations = [pybullet.getQuaternionFromEuler([0, 0, 0]) for _ in range(RobotTeam.NUM_ROBOTS)]
    robot_controller.move_robots_simultaneously(list(zip(initial_robot_pos, initial_orientations)))

    # Record frames after robot positioning
    for _ in range(10):
        pybullet.stepSimulation()
        recorder.capture_frame(box_id)

    # Get robot commands and transformation data
    data = get_robot_commands(box_size, box_position, np.array([0, 0, 0]), initial_robot_pos)
    robot_commands = data['commands']
    shifts = data['shifts']
    rotation_vectors = data['rotation_vectors']
    initial_point = data['initial_point']['intersection_point']
    
    # Dictionary to store robot constraints
    robot_constraints = [None] * RobotTeam.NUM_ROBOTS
    
    # Track current shift index
    shift_index = 1
    
    # Process each timestep
    shift_counter = 0
    for step_idx, timestep in enumerate(robot_commands):
        print(f"\nTimestep {step_idx}:")
        
        # Determine if this is a shift action
        is_shift = any(timestep[i]['action'] == "shift" for i in range(RobotTeam.NUM_ROBOTS))
        
        # Release robot constraints
        for i in range(RobotTeam.NUM_ROBOTS):
            if robot_constraints[i] is not None:
                pybullet.removeConstraint(robot_constraints[i])
                robot_constraints[i] = None
        
        if is_shift:
            shift_counter += 1
            print(f"SHIFT ACTION {shift_index} - Moving box using predefined transformations")
            
            # Remove box constraint
            if box_constraint:
                pybullet.removeConstraint(box_constraint)
                box_constraint = None
            
            # Get the current box position and orientation
            current_pos, current_orn = pybullet.getBasePositionAndOrientation(box_id)
            
            # Calculate target position and orientation using shifts and rotation_vectors
            if shift_index < len(shifts):
                # Apply the shift
                shift = shifts[shift_index]
                rotation = rotation_vectors[shift_index]
                
                # Create rotation matrix from Euler angles
                rotation_matrix = Rotation.from_euler('xyz', rotation).as_matrix()
                
                # Rotate around the intersection point and apply shift
                target_position = rotation_matrix @ (box_position-initial_point) + initial_point - rotation_matrix@shift
                target_orientation = rotation
                
                print(f"Shift vector: {shift}")
                print(f"Rotation: {rotation * 180/np.pi} degrees")
                
                # Increment shift index for next time
                shift_index += 1
            
            # Prepare robot positions for movement
            robot_poses = []
            for robot_idx in range(RobotTeam.NUM_ROBOTS):
                cmd = timestep[robot_idx]
                pos = cmd['position']
                ori = cmd['orientation']
                
                if isinstance(ori, (list, tuple, np.ndarray)) and len(ori) == 3:
                    ori = pybullet.getQuaternionFromEuler(ori)
                    
                robot_poses.append((pos, ori))
                print(f"Robot {robot_idx}: Moving to new position for shift")
            
            # Move robots to their positions - do this first
            robot_controller.move_robots_simultaneously(robot_poses, vel)
            
            # Record a few frames after robot movement
            for _ in range(10):
                pybullet.stepSimulation()
                recorder.capture_frame(box_id)
                
            # Move the box to new position - do this second
            if isinstance(target_orientation, (list, tuple, np.ndarray)) and len(target_orientation) == 3:
                target_orientation_quat = pybullet.getQuaternionFromEuler(target_orientation)
            else:
                target_orientation_quat = target_orientation
                
            pybullet.resetBasePositionAndOrientation(box_id, target_position, target_orientation_quat)
            pybullet.stepSimulation()
            
            # Record a few frames after box movement
            for _ in range(10):
                pybullet.stepSimulation()
                recorder.capture_frame(box_id)
                
            # Create new constraint for box at its new position
            box_constraint = pybullet.createConstraint(
                parentBodyUniqueId=box_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=target_position,
                childFrameOrientation=target_orientation_quat
            )
            
        else:  # Step or other action
            # Handle non-shift actions - only move robots
            robot_poses = []
            for robot_idx in range(RobotTeam.NUM_ROBOTS):
                cmd = timestep[robot_idx]
                pos = cmd['position']
                ori = cmd['orientation']
                
                if isinstance(ori, (list, tuple, np.ndarray)) and len(ori) == 3:
                    ori = pybullet.getQuaternionFromEuler(ori)
                    
                robot_poses.append((pos, ori))
                print(f"Robot {robot_idx}: Action={cmd['action']}")
            
            # Move robots to their positions
            robot_controller.move_robots_simultaneously(robot_poses, vel)
            
            # Record frames during robot movement
            for _ in range(10):
                pybullet.stepSimulation()
                recorder.capture_frame(box_id)
        
        # Allow physics to settle and capture frames
        for _ in range(10):
            pybullet.stepSimulation()
            recorder.capture_frame(box_id)

    # Capture some final frames
    for _ in range(30):
        pybullet.stepSimulation()
        recorder.capture_frame(box_id)

    # Stop recording
    recorder.stop()

    # Clean up constraints at the end
    if box_constraint:
        pybullet.removeConstraint(box_constraint)
        
    for i in range(RobotTeam.NUM_ROBOTS):
        if robot_constraints[i]:
            pybullet.removeConstraint(robot_constraints[i])

    print("\nMotion sequence completed and video saved")
    time.sleep(5)


if __name__ == "__main__":
    main()