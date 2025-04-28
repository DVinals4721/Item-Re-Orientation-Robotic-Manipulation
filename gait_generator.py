from path_generator import PrismSurfacePathfinder
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment

class Step:
    def __init__(self):
        self.action_type = "step"
        self.points = []
        self.robot_id = None
        self.suction_actions = []

class Shift:
    def __init__(self):
        self.action_type = "shift"
        self.robot_points = []
        self.suction_actions = []

def get_rotation_matrix_and_euler(z_axis):
    # Normalize z vector
    z_axis = z_axis / np.linalg.norm(z_axis)

    # To get 0 yaw, we want the x axis to be in the world XZ plane
    # First, project [1,0,0] onto the plane perpendicular to z_axis
    x_axis = np.array([1.0, 0.0, 0.0])
    proj = np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis - proj

    # Normalize x_axis
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Get y_axis using cross product
    y_axis = np.cross(z_axis, x_axis)

    # Form rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Get Euler angles (in radians)
    euler_angles = np.array([
        np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]),  # roll
        np.arcsin(-rotation_matrix[2,0]),                        # pitch
        np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])   # yaw
    ])

    return rotation_matrix, euler_angles

def create_robot_commands(gait, robot_assignment, initial_robot_positions):
    robot_commands = []
    current_positions = {}
    current_normals = {}
    current_suctions = {}

    # Create initial command to move to first positions
    initial_command = {}
    # Find the first step action for each robot
    for orig_robot_id, path_id in robot_assignment.items():
        # Look for the first step action that controls this robot's path
        for action in gait[:5]:
            if action.action_type == "step" and action.robot_id == path_id and action.points:
                # Found a step action for this robot's path
                point = action.points[0]  # Use the first point of the step
                normal = np.array(point['normal'])
                R, euler = get_rotation_matrix_and_euler(normal)

                initial_command[orig_robot_id] = {
                    'position': point['position'],
                    'orientation': euler,
                    'suction': 0,  # Start with suction off
                    'next_suction': 1,
                    'action':"initial",
                    'robot':"all"
                }

                # Initialize current states
                current_positions[orig_robot_id] = point['position']
                current_normals[orig_robot_id] = point['normal']
                current_suctions[orig_robot_id] = 1

                # Found what we need for this robot, move to the next
                break

    # Add initial command as first timestep
    robot_commands.append(initial_command)

    # Create reverse mapping from path_id to original robot_id
    path_to_robot = {v: k for k, v in robot_assignment.items()}

    for i, action in enumerate(gait):
        timestep_command = {}

        # Look ahead for next suction command
        next_suction = {}
        if i + 1 < len(gait):
            next_action = gait[i + 1]
            if next_action.action_type == "step":
                for orig_robot_id in current_suctions.keys():
                    path_id = robot_assignment[orig_robot_id]
                    if path_id == next_action.robot_id:
                        next_suction[orig_robot_id] = next_action.suction_actions[0]
                    else:
                        next_suction[orig_robot_id] = current_suctions[orig_robot_id]
            else:  # shift
                for orig_robot_id in current_suctions.keys():
                    path_id = robot_assignment[orig_robot_id]
                    next_suction[orig_robot_id] = next_action.suction_actions[path_id]
        else:
            next_suction = current_suctions.copy()

        if action.action_type == "step":
            path_id = action.robot_id
            moving_robot = path_to_robot[path_id]

            for i, point in enumerate(action.points):
                command_dict = {}

                # Commands for the moving robot
                normal = np.array(point['normal'])
                R, euler = get_rotation_matrix_and_euler(normal)
                command_dict[moving_robot] = {
                    'position': point['position'],
                    'orientation': euler,
                    'suction': action.suction_actions[i],
                    'next_suction': next_suction[moving_robot],
                    'action':"step",
                    'robot':str(path_id)
                }

                # Commands for stationary robots
                for orig_robot_id in current_positions.keys():
                    if orig_robot_id != moving_robot:
                        normal = np.array(current_normals[orig_robot_id])
                        R, euler = get_rotation_matrix_and_euler(normal)
                        command_dict[orig_robot_id] = {
                            'position': current_positions[orig_robot_id],
                            'orientation': euler,
                            'suction': current_suctions[orig_robot_id],
                            'next_suction': next_suction[orig_robot_id],
                            'action':"stay",
                            'robot':str(path_id)
                        }

                robot_commands.append(command_dict)

            # Update current states for moving robot
            current_positions[moving_robot] = action.points[-1]['position']
            current_normals[moving_robot] = action.points[-1]['normal']
            current_suctions[moving_robot] = action.suction_actions[-1]

        elif action.action_type == "shift":
            command_dict = {}

            for path_id, point in enumerate(action.robot_points):
                orig_robot_id = path_to_robot[path_id]
                normal = np.array(point['normal'])
                R, euler = get_rotation_matrix_and_euler(normal)
                command_dict[orig_robot_id] = {
                    'position': point['position'],
                    'orientation': euler,
                    'suction': action.suction_actions[path_id],
                    'next_suction': next_suction[orig_robot_id],
                    'action':"shift",
                    'robot':"All"

                }

                current_positions[orig_robot_id] = point['position']
                current_normals[orig_robot_id] = point['normal']
                current_suctions[orig_robot_id] = action.suction_actions[path_id]

            robot_commands.append(command_dict)

    return robot_commands

def plot_gait_steps(gait, num_steps_to_plot=None, plot_all_steps=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if num_steps_to_plot is None:
        num_steps_to_plot = len(gait)

    robot_colors = ['r', 'g', 'b', 'y']
    shift_colormap = cm.rainbow

    last_positions = {}
    shift_count = 0

    for i, action in enumerate(gait[:num_steps_to_plot]):
        if action.action_type == "step":
            points = np.array([point['position'] for point in action.points])
            robot_id = action.robot_id
            color = robot_colors[robot_id % len(robot_colors)]

            if plot_all_steps:
                # Plot all intermediate points
                ax.plot(points[:, 0], points[:, 1], points[:, 2],
                       marker='o', color=color, label=f'Robot {robot_id}',
                       linestyle='-', linewidth=2, alpha=0.3)
            else:
                # Plot only start and end points
                ax.plot([points[0, 0], points[-1, 0]],
                       [points[0, 1], points[-1, 1]],
                       [points[0, 2], points[-1, 2]],
                       marker='o', color=color, label=f'Robot {robot_id}',
                       linestyle='-', linewidth=2)

            last_positions[robot_id] = points[-1]
        elif action.action_type == "shift":
            if isinstance(action.robot_points[0], dict):
                points = np.array([point['position'] for point in action.robot_points])
            shift_color = shift_colormap(0.1 + (shift_count * 0.2) % 0.9)

            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      marker='x', color=shift_color, s=100, label=f'Shift {shift_count+1}')

            for robot_id in range(len(points)):
                if robot_id in last_positions:
                    line_points = np.vstack((last_positions[robot_id], points[robot_id]))
                    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                           color=shift_color, linestyle='--', linewidth=1.5)

            for robot_id in range(len(points)):
                last_positions[robot_id] = points[robot_id]

            shift_count += 1

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Gait Visualization (First {num_steps_to_plot} steps)')
    ax.set_box_aspect([1,1,1])
    ax.grid(True)

    plt.show()
class GaitGenerator:
    def __init__(self, dimensions, resolution, clearance_radius, num_arc_points,min_intersection_distance=0,arc_height_factor=1,orientation_factor = None):
        self.pathfinder = PrismSurfacePathfinder(dimensions, resolution, clearance_radius, num_arc_points,min_intersection_distance,arc_height_factor)
        self.initial_point = None
        self.paths = None
        self.gait = []
        self.robot_commands = None
        self.robot_assignment = None
        self.orientation_factor = orientation_factor
    def plot_trajectory(self):
        self.pathfinder.plot_prism_surface_points()
    def generate_paths(self, initial_centroid, initial_orientation, target_orientation):
        self.paths, self.initial_point = self.pathfinder.find_trajectory(
            initial_centroid, initial_orientation, target_orientation)

        self.shifts = self._get_shifts()
        self.rotation_vectors = self._get_rotation_vectors(initial_orientation, target_orientation)

        self._generate_gait()

    def assign_robots(self, initial_robot_positions):
        self.robot_assignment = self._get_robot_assignment(initial_robot_positions)
        self.robot_commands = self._create_robot_commands(initial_robot_positions)
        return self.robot_commands

    def _get_shifts(self):
        original_indices = [i for i, p in enumerate(self.paths[0]) if p['type'] == "original"]
        centroids = []
        for index in original_indices:
            og_point = [path[index]['position'] for path in self.paths]
            centroid = np.mean(og_point, axis=0)
            centroids.append(centroid)

        self.centroids = centroids
        return centroids - centroids[0]


    def _get_rotation_vectors(self, initial_orientation, final_orientation, base=30):
        num_vectors = len(self.shifts)
        base = self.orientation_factor

        # Create exponentially spaced values between 0 and 1
        t1 = (np.power(base[1], np.linspace(0, 1, num_vectors)) - 1) / (base[1] - 1)
        t2 = (np.power(base[0], np.linspace(0, 1, num_vectors)) - 1) / (base[0] - 1)

        # Interpolate between initial and final orientations using the exponential spacing
        roll = initial_orientation[0] + t1 * (-final_orientation[0] - initial_orientation[0])
        pitch = initial_orientation[1] + t2 * (-final_orientation[1] - initial_orientation[1])
        yaw = initial_orientation[2] + t1 * (-final_orientation[2] - initial_orientation[2])

        return np.column_stack([roll, pitch, yaw])

    def _get_new_surface_point(self, surface_point, index):
        apply_shift = self.shifts[index]
        Rot = Rotation.from_euler('xyz', self.rotation_vectors[index]).as_matrix()
        i_point = np.array(self.initial_point['intersection_point'])
        return Rot @ (surface_point-i_point) + i_point - Rot@apply_shift

    def _generate_gait(self):
        self.gait = []
        original_indices = [i for i, p in enumerate(self.paths[0]) if p['type'] == "original"]
        self.rotations_mats = []

        for vectors in self.rotation_vectors:
            self.rotations_mats.append(Rotation.from_euler('xyz', vectors).as_matrix())


        for i, (start, end) in enumerate(zip(original_indices, original_indices[1:])):
            shift = Shift()
            shift_pos = []

            for p, path in enumerate(self.paths):
                step = Step()
                step_points = []

                for point in path[start:end+1]:
                    point_new = point.copy()
                    point_new['position'] = self._get_new_surface_point(point_new['position'], i)
                    point_new['normal'] = np.dot(Rotation.from_euler('xyz', self.rotation_vectors[i]).as_matrix(),
                                               point_new['normal'])
                    step_points.append(point_new)

                step.points = step_points
                step.robot_id = p
                step.suction_actions = [1]+[0]*len(path[start+1:end])+[1]

                point_shift = path[end].copy()
                point_shift['position'] = self._get_new_surface_point(point_shift['position'], i+1)
                point_shift['normal'] = np.dot(Rotation.from_euler('xyz', self.rotation_vectors[i+1]).as_matrix(),
                                            point_shift['normal'])
                shift_pos.append(point_shift)
                self.gait.append(step)

            shift.robot_points = shift_pos
            shift.suction_actions = [1]*len(self.paths)
            self.gait.append(shift)

    def _get_robot_assignment(self, initial_robot_positions):
        num_robots = len(initial_robot_positions)
        first_points = [path[0]['position'] for path in self.paths]

        distances = np.zeros((num_robots, num_robots))
        for i, robot_pos in enumerate(initial_robot_positions):
            for j, path_point in enumerate(first_points):
                distances[i,j] = np.linalg.norm(np.array(robot_pos) - np.array(path_point))

        robot_indices, path_indices = linear_sum_assignment(distances)
        return dict(zip(robot_indices, path_indices))

    def _create_robot_commands(self, initial_robot_positions):
        return create_robot_commands(self.gait, self.robot_assignment, initial_robot_positions)

    def plot_gait(self, num_steps=None, plot_all_steps=False):
        plot_gait_steps(self.gait, num_steps, plot_all_steps)

# Example usage:
if __name__ == "__main__":
    dimensions = np.array([10, 5, 5])
    resolution = 0.5
    clearance_radius = 1
    num_arc_points = 4

    initial_robot_positions = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ]

    # Create generator
    generator=GaitGenerator( dimensions, resolution, clearance_radius, num_arc_points,0.1,2,orientation_factor=10*dimensions)

    # Generate paths
    generator.generate_paths(
        initial_centroid=np.array([0, 0, 0]),
        initial_orientation=np.array([0, 0, 0]),
        target_orientation=np.array([0, np.pi/2, 0])
    )

    for action in generator.gait:
        if action.action_type == "step":
            print("step")
            for point,suction in zip(action.points,action.suction_actions):
                robot_id =action.robot_id
                pos = point['position']
                normal = point['normal']
                print(normal)
                suction_command = suction

        if action.action_type == "shift":
            print("shift")
            for i,point in enumerate(action.robot_points):
                robot_id =i
                pos = point['position']
                normal = point['normal']
                print(normal)
    #print(generator.gait[1].points[0]['position'])
    #generator.plot_trajectory()

    """for rotations in generator.rotations_mats:
        print(rotations)"""

    # Get robot commands
    robot_commands = generator.assign_robots(initial_robot_positions)
    for i, timestep in enumerate(robot_commands):
        print(f"\nTimestep {i}:")
        for robot_id, command in timestep.items():
            print(f"Robot {robot_id}:")
            print(f"  Position: {command['position']}")
            print(f"  Orientation: {command['orientation']}")
            print(f"  Current Suction: {command['suction']}")
            print(f"  Next Suction: {command['next_suction']}")
    # Optional: visualize the gait
    generator.plot_trajectory()
    generator.plot_gait(plot_all_steps=True)
