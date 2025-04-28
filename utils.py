# utils.py
import pybullet
import math
import numpy as np
from scipy.spatial.transform import Rotation

def get_rotation_matrix(orientation):
    return pybullet.getMatrixFromQuaternion(orientation)

def visualize_frame(position, orientation, line_length=0.1, line_width=2, lifetime=0.1):
    rot_matrix = get_rotation_matrix(orientation)
    for i, color in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        axis = [rot_matrix[i], rot_matrix[i+3], rot_matrix[i+6]]
        end_point = [position[j] + line_length * axis[j] for j in range(3)]
        pybullet.addUserDebugLine(position, end_point, color, lineWidth=line_width, lifeTime=lifetime)

def generate_circular_points(center, radius, num_points, height):
    return [[center[0] + radius * math.cos(2 * math.pi * i / num_points),
             center[1] + radius * math.sin(2 * math.pi * i / num_points),
             height] for i in range(num_points)]
def generate_random_orientation(max_angle):
    # Generate a random rotation axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normalize to unit vector

    # Generate a random angle within the current curriculum stage
    angle = np.random.uniform(-max_angle, max_angle)

    # Create a rotation object
    r = Rotation.from_rotvec(angle * axis)

    # Convert to quaternion
    quat = r.as_quat()

    return tuple(quat)
def calculate_box_angle_error(current_orientation, target_orientation):
    # Assuming orientations are quaternions
    current_quat = np.array(current_orientation)
    target_quat = np.array(target_orientation)

    # Calculate the difference quaternion
    diff_quat = quaternion_multiply(quaternion_inverse(current_quat), target_quat)

    # Convert to axis-angle representation
    angle = 2 * np.arccos(diff_quat[0])

    return np.degrees(angle)  # Return the error in degrees

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)
