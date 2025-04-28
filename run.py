# showcase.py
from config import SimulationConfig, RobotTeam, Item, RobotConfig
import pybullet
import pybullet_data
import numpy as np
import math
from robot import UR5Robot
import random
import time
from robot_controller import RobotController
from utils import generate_circular_points
from simulation import SimulationEnvironment
from robot_controller import RobotController
import utils
from gait_generator import GaitGenerator
from scipy.spatial.transform import Rotation
import threading

resolution = 0.05
clearance_radius = 0.01
num_arc_points = 4
min_intersection_distance = 0.2
arc_height_factor=2
orien_progress_factor = 10
box_size = np.array([2,1,1])
orien_progress = box_size*orien_progress_factor
#goal_orientation = np.array([np.pi/2,0, 0])
goal_orientation = np.array([0,np.pi/2, 0])
vel = 10


def get_robot_commands(box_size, box_position, box_orientation, initial_robot_pos):
    generator = GaitGenerator( box_size, resolution, clearance_radius, num_arc_points,min_intersection_distance,arc_height_factor,orientation_factor=orien_progress)
    # Generate paths
    generator.generate_paths(
        initial_centroid=np.array(box_position),
        initial_orientation=box_orientation,
        target_orientation=goal_orientation
    )

    # Get robot commands
    robot_commands = generator.assign_robots(initial_robot_pos)
    #generator.plot_trajectory()
    #generator.plot_gait(plot_all_steps=True)
    # Return additional transformation data for box movement
    return {
        'commands': robot_commands,
        'shifts': generator.shifts,
        'rotation_vectors': generator.rotation_vectors,
        'initial_point': generator.initial_point
    }

def move_box_thread(box_id, position, orientation, box_constraint=None, smooth=True, steps=100):
    """
    Function to move a box to a new position and orientation, and constrain it in that pose

    Args:
        box_id: PyBullet ID of the box object
        position: Target position [x, y, z]
        orientation: Target orientation (Euler angles or quaternion)
        box_constraint: Current constraint ID if one exists
        smooth: Whether to move smoothly or instantly
        steps: Number of steps for smooth movement

    Returns:
        New constraint ID
    """
    # Convert orientation to quaternion if it's Euler angles
    if isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 3:
        orientation_quat = pybullet.getQuaternionFromEuler(orientation)
    else:
        orientation_quat = orientation

    # Get current position and orientation
    current_pos, current_orn = pybullet.getBasePositionAndOrientation(box_id)

    print(f"Moving box from {current_pos} to {position}")

    # Remove existing constraint if provided
    if box_constraint is not None:
        pybullet.removeConstraint(box_constraint)
        box_constraint = None

    if not smooth:
        # Instant movement
        pybullet.resetBasePositionAndOrientation(box_id, position, orientation_quat)
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
            interp_orn = pybullet.getQuaternionSlerp(current_orn, orientation_quat, t)

            # Apply the interpolated pose
            pybullet.resetBasePositionAndOrientation(box_id, interp_pos, interp_orn)
            pybullet.stepSimulation()

    # Create new constraint to hold box in place
    new_constraint = pybullet.createConstraint(
        parentBodyUniqueId=box_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=pybullet.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=position,
        childFrameOrientation=orientation_quat
    )

    return new_constraint

def main():
    sim_env = SimulationEnvironment(goal_orientation, use_gui=True)
    robots = sim_env.robots
    robot_controller = RobotController(sim_env.robots)

    # Generate box
    box_position = np.array([Item.X_Y_CENTER[0], Item.X_Y_CENTER[1],
                           RobotTeam.CENTER[2] - Item.HEIGHT_FROM_ROBOT - box_size[2]/2])
    box_id = sim_env.spawn_box(box_position, box_size, Item.MASS, Item.COLOR, "main_box")

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
        box_position, radius=0.3,
        num_points=RobotTeam.NUM_ROBOTS, height=box_position[2] + box_size[2]/2 + 0.01
    )
    initial_orientations = [pybullet.getQuaternionFromEuler([0, 0, 0]) for _ in range(RobotTeam.NUM_ROBOTS)]
    robot_controller.move_robots_simultaneously(list(zip(initial_robot_pos, initial_orientations)))

    # Get robot commands and transformation data
    data = get_robot_commands(box_size, box_position, np.array([0, 0, 0]), initial_robot_pos)
    robot_commands = data['commands']
    shifts = data['shifts']
    rotation_vectors = data['rotation_vectors']
    initial_point = data['initial_point']['intersection_point']

    # Track current shift index
    shift_index = 1
    shift_counter = 0

    # Process each timestep
    for step_idx, timestep in enumerate(robot_commands):
        print(f"\nTimestep {step_idx}:")

        # Determine if this is a shift action
        is_shift = any(timestep[i]['action'] == "shift" for i in range(RobotTeam.NUM_ROBOTS))

        if is_shift:
            shift_counter += 1
            print(f"SHIFT ACTION {shift_index} - Moving box using predefined transformations")

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

            # Move the box with current constraint and get new constraint
            box_constraint = move_box_thread(
                box_id,
                target_position,
                target_orientation,
                box_constraint,
                smooth=False,
                steps=1  # No need for steps since smooth=False
            )

            # Move robots to their positions
            robot_controller.move_robots_simultaneously(robot_poses, vel)

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

        # Allow physics to settle
        for _ in range(10):
            pybullet.stepSimulation()

    # Clean up constraint at the end
    if box_constraint:
        pybullet.removeConstraint(box_constraint)

    print("\nMotion sequence completed")
    time.sleep(60)


if __name__ == "__main__":
    main()