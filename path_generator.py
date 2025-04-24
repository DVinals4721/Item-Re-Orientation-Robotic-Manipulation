import numpy as np
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from collections import deque
import heapq
from itertools import product
from matplotlib.colors import ListedColormap

class PrismSurfacePathfinder:
    def __init__(self, dimensions, resolution, clearance_radius, num_arc_points=5, min_intersection_distance=0, arc_height_factor=1.0):
        """
        Initialize the PrismSurfacePathfinder class.
        
        Parameters:
        dimensions (np.array): [length, width, height] of the prism
        resolution (float): Spacing between points
        clearance_radius (float): Radius to maintain clearance around edge points
        num_arc_points (int): Number of arc points to add between path segments
        min_intersection_distance (float): Minimum distance between intersection point and selected points
        arc_height_factor (float): Factor controlling the height of arcs between points (1.0 = standard semicircular arc)
        """
        self.dimensions = dimensions
        self.resolution = resolution
        self.clearance_radius = clearance_radius
        self.num_arc_points = num_arc_points
        self.min_intersection_distance = min_intersection_distance
        self.arc_height_factor = arc_height_factor  # New parameter for arc height control
        self.surface_points = None
        self.original_result = None
        self.rotated_result = None
        self.paths = None
        self.processed_paths = None
    
    def create_prism_surface_points(self, centroid, orientation):
        """
        Create a data point bank for the surface of a rectangular prism with consistent normals
        and only direct neighbors, labeling points as face or edge points.

        Parameters:
        centroid (np.array): [x, y, z] coordinates of the prism's center
        orientation (np.array): [rx, ry, rz] rotation angles in radians

        Returns:
        list: List of dictionaries, each containing point data
        """
        length, width, height = self.dimensions
        rotation_matrix = Rotation.from_euler('xyz', orientation).as_matrix()

        # Generate points on each face
        faces = [
            ('x', np.array([1, 0, 0]), np.linspace(-length/2, length/2, int(length/self.resolution))),
            ('y', np.array([0, 1, 0]), np.linspace(-width/2, width/2, int(width/self.resolution))),
            ('z', np.array([0, 0, 1]), np.linspace(-height/2, height/2, int(height/self.resolution)))
        ]

        points = {}
        for axis, normal, line in faces:
            for sign in [-1, 1]:
                if axis == 'x':
                    y, z = np.meshgrid(np.linspace(-width/2, width/2, int(width/self.resolution)),
                                       np.linspace(-height/2, height/2, int(height/self.resolution)))
                    x = np.full_like(y, sign * length/2)
                elif axis == 'y':
                    x, z = np.meshgrid(np.linspace(-length/2, length/2, int(length/self.resolution)),
                                       np.linspace(-height/2, height/2, int(height/self.resolution)))
                    y = np.full_like(x, sign * width/2)
                else:  # axis == 'z'
                    x, y = np.meshgrid(np.linspace(-length/2, length/2, int(length/self.resolution)),
                                       np.linspace(-width/2, width/2, int(width/self.resolution)))
                    z = np.full_like(x, sign * height/2)

                face_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
                rotated_points = np.dot(face_points, rotation_matrix.T) + centroid
                rotated_normal = np.dot(sign * normal, rotation_matrix.T)

                for point in rotated_points:
                    point_tuple = tuple(np.round(point, 6))  # Round to avoid floating point issues
                    if point_tuple not in points:
                        points[point_tuple] = {
                            'position': point,
                            'normal': rotated_normal,
                            'neighbors': set(),
                            'type': 'face'  # Initially label all points as face points
                        }
                    elif axis == 'x':  # Prioritize x-normal
                        points[point_tuple]['normal'] = rotated_normal

        # Convert points dict to list and add IDs
        points_list = []
        for i, (point_tuple, point_data) in enumerate(points.items()):
            point_data['id'] = i
            points_list.append(point_data)

        # Create a KD-tree for efficient nearest neighbor search
        tree = cKDTree(np.array([p['position'] for p in points_list]))

        # Find direct neighbors and identify edge points
        for point in points_list:
            edge_dimensions = 0
            for dim, size in zip(point['position'] - centroid, self.dimensions/2):
                if np.isclose(abs(dim), size, atol=self.resolution/2):
                    edge_dimensions += 1
            
            # Find neighbors within 1.5 * resolution distance
            neighbors = tree.query_ball_point(point['position'], 1.5 * self.resolution)
            
            for neighbor_idx in neighbors:
                if neighbor_idx != point['id']:
                    diff = np.round(point['position'] - points_list[neighbor_idx]['position'], 6)
                    if np.count_nonzero(diff) == 1 and np.any(np.abs(diff) <= self.resolution):
                        point['neighbors'].add(neighbor_idx)
            
            if edge_dimensions >= 2:
                point['type'] = 'edge'

        # Convert neighbor sets to lists for JSON serialization
        for point in points_list:
            point['neighbors'] = list(point['neighbors'])

        # Debug print
        face_count = sum(1 for point in points_list if point['type'] == 'face')
        edge_count = sum(1 for point in points_list if point['type'] == 'edge')
        print(f"Created {face_count} face points and {edge_count} edge points")

        # Additional debug information
        avg_neighbors = sum(len(p['neighbors']) for p in points_list) / len(points_list)
        print(f"Average number of neighbors per point: {avg_neighbors:.2f}")
        min_neighbors = min(len(p['neighbors']) for p in points_list)
        max_neighbors = max(len(p['neighbors']) for p in points_list)
        print(f"Min neighbors: {min_neighbors}, Max neighbors: {max_neighbors}")

        self.surface_points = points_list
        return points_list

    def transform_prism_points(self, surface_points, new_centroid, new_orientation, original_centroid, original_orientation):
        """
        Transform the points and normals of the prism based on a new centroid position and orientation.

        Parameters:
        surface_points (list): List of dictionaries containing point data
        new_centroid (np.array): New [x, y, z] coordinates of the prism's center
        new_orientation (np.array): New [rx, ry, rz] rotation angles in radians
        original_centroid (np.array): Original [x, y, z] coordinates of the prism's center
        original_orientation (np.array): Original [rx, ry, rz] rotation angles in radians

        Returns:
        list: List of dictionaries with transformed point data
        """
        # Calculate rotation matrices
        original_rotation = Rotation.from_euler('xyz', original_orientation)
        new_rotation = Rotation.from_euler('xyz', new_orientation)

        # Calculate the transformation matrix
        transform_rotation = new_rotation * original_rotation.inv()

        transformed_points = []
        for point in surface_points:
            # Transform position
            original_position = point['position'] - original_centroid
            rotated_position = transform_rotation.apply(original_position)
            new_position = rotated_position + new_centroid

            # Transform normal
            new_normal = transform_rotation.apply(point['normal'])

            transformed_points.append({
                'id': point['id'],
                'position': new_position,
                'normal': new_normal,
                'neighbors': point['neighbors']
            })

        return transformed_points

    def ray_box_intersection(self, ray_origin, ray_direction, box_min, box_max):
        """
        Calculate the intersection point of a ray with an axis-aligned box.
        
        Parameters:
        ray_origin (np.array): Origin of the ray
        ray_direction (np.array): Direction of the ray (normalized)
        box_min (np.array): Minimum corner of the box
        box_max (np.array): Maximum corner of the box
        
        Returns:
        np.array or None: Intersection point if it exists, None otherwise
        """
        t_min = (box_min - ray_origin) / ray_direction
        t_max = (box_max - ray_origin) / ray_direction
        
        t_near = np.minimum(t_min, t_max)
        t_far = np.maximum(t_min, t_max)
        
        t_nearest = np.max(t_near)
        t_farthest = np.min(t_far)
        
        if t_nearest <= t_farthest and t_farthest > 0:
            return ray_origin + t_nearest * ray_direction
        
        return None

    def find_surface_points(self, orientation):
        """
        Find 4 surface points based on a rotated vector and clearance requirements,
        relative to the initial pose of the geometry, avoiding spheres intersecting with edge points
        and with each other, and shifting points to intersect the vector.
        Points will maintain a minimum specified distance from the intersection point.
        """
        if self.surface_points is None:
            raise ValueError("Surface points haven't been created yet. Call create_prism_surface_points first.")
                
        surface_points = self.surface_points
        positions = np.array([p['position'] for p in surface_points])
        normals = np.array([p['normal'] for p in surface_points])
        
        initial_vector = np.array([0, 0, 1])
        rotation = Rotation.from_euler('xyz', orientation)
        rotated_vector = rotation.apply(initial_vector)
        
        center = np.mean(positions, axis=0)
        half_dimensions = self.dimensions / 2
        box_min = center - half_dimensions
        box_max = center + half_dimensions
        
        intersection_point = self.ray_box_intersection(center, -rotated_vector, box_min, box_max)
        
        if intersection_point is None:
            raise ValueError("Ray does not intersect the prism")
        
        tree = cKDTree(positions)
        
        # Find the closest point to the intersection point and use its normal
        _, closest_idx = tree.query(intersection_point)
        intersection_normal = normals[closest_idx]
        
        # Define the corners of the imaginary square - ensure size respects min_intersection_distance
        square_size = max(self.clearance_radius * 2, self.min_intersection_distance * 2)
        square_corners = np.array([
            [-1, -1, 0],
            [-1,  1, 0],
            [ 1, -1, 0],
            [ 1,  1, 0]
        ]) * (square_size / 2)
        
        # Create a local coordinate system
        z_axis = intersection_normal
        x_axis = np.cross(z_axis, [0, 0, 1])
        if np.allclose(x_axis, 0):
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Create the transformation matrix
        local_to_global = np.column_stack((x_axis, y_axis, z_axis))
        
        # Transform square corners to 3D space
        corners_3d = intersection_point + np.dot(square_corners, local_to_global.T)
        
        # Helper function to check if point is far enough from intersection
        def is_far_enough_from_intersection(point):
            return np.linalg.norm(point['position'] - intersection_point) >= self.min_intersection_distance
        
        def sphere_intersects_edge(point, radius):
            _, indices = tree.query(point['position'], k=len(surface_points), distance_upper_bound=radius)
            return any(surface_points[i]['type'] == 'edge' for i in indices if i < len(surface_points))
        
        def spheres_intersect(p1, p2, radius):
            return np.linalg.norm(p1['position'] - p2['position']) < 2 * radius
        
        chosen_points = []
        for corner in corners_3d:
            # Get all points sorted by distance to the corner
            distances, indices = tree.query(corner, k=len(surface_points))
            
            for idx in indices:
                point = surface_points[idx]
                
                # Check minimum distance from intersection point
                if not is_far_enough_from_intersection(point):
                    continue
                    
                # Check if the sphere around this point intersects any edge point or other chosen points
                if not sphere_intersects_edge(point, self.clearance_radius) and \
                not any(spheres_intersect(point, cp, self.clearance_radius) for cp in chosen_points):
                    chosen_points.append(point)
                    break
            
            if len(chosen_points) == 4:
                break
        
        # If we couldn't find 4 non-intersecting points, relax the edge intersection constraint
        # but still maintain minimum distance from intersection
        if len(chosen_points) < 4:
            for corner in corners_3d[len(chosen_points):]:
                distances, indices = tree.query(corner, k=len(surface_points))
                
                for idx in indices:
                    point = surface_points[idx]
                    
                    # Still maintain minimum distance constraint
                    if not is_far_enough_from_intersection(point):
                        continue
                        
                    if not any(spheres_intersect(point, cp, self.clearance_radius) for cp in chosen_points):
                        chosen_points.append(point)
                        break
                
                if len(chosen_points) == 4:
                    break
        
        # In the unlikely event that we still didn't find 4 points, 
        # find the farthest points that meet the distance requirement
        while len(chosen_points) < 4:
            remaining_indices = set(range(len(surface_points))) - set(p['id'] for p in chosen_points)
            if not remaining_indices:
                break
                
            # Find farthest remaining point that satisfies minimum distance
            valid_points = [(i, np.linalg.norm(surface_points[i]['position'] - intersection_point)) 
                        for i in remaining_indices 
                        if np.linalg.norm(surface_points[i]['position'] - intersection_point) >= self.min_intersection_distance]
            
            if valid_points:
                # Sort by distance (descending)
                valid_points.sort(key=lambda x: x[1], reverse=True)
                farthest_idx = valid_points[0][0]
            else:
                # If no points meet the minimum distance, take the farthest available point
                farthest_idx = max(remaining_indices, 
                                key=lambda i: np.linalg.norm(surface_points[i]['position'] - intersection_point))
            
            chosen_points.append(surface_points[farthest_idx])
        
        # Shift points to intersect the vector
        chosen_positions = np.array([p['position'] for p in chosen_points])
        average_point = np.mean(chosen_positions, axis=0)
        
        # Calculate the shift vector
        shift_vector = intersection_point - average_point
        
        # Project the shift vector onto the plane perpendicular to the normal
        shift_vector = shift_vector - np.dot(shift_vector, intersection_normal) * intersection_normal
        
        # Apply the shift to all chosen points
        for point in chosen_points:
            point['position'] += shift_vector
        
        result = {
            'intersection_point': intersection_point,
            'closest_points': chosen_points,
            'rotated_vector': rotated_vector,
            'center': center,
            'intersection_normal': intersection_normal,
            'shift_vector': shift_vector
        }
        
        return result

    def vector_intersects(self, v1_start, v1_end, v2_start, v2_end):
        """Check if two vectors intersect in 2D space."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(v1_start, v2_start, v2_end) != ccw(v1_end, v2_start, v2_end) and \
               ccw(v1_start, v1_end, v2_start) != ccw(v1_start, v1_end, v2_end)

    def find_matching_pairs(self, original_points, new_points):
        """
        Find pairs of points whose vectors do not intersect the average vector,
        then select the middle 4 pairs.
        
        Parameters:
        original_points (list): List of dictionaries containing the original point data.
        new_points (list): List of dictionaries containing the new (translated) point data.
        
        Returns:
        list: List of tuples (original_index, new_index) representing the selected pairs.
        """
        if len(original_points) != len(new_points):
            raise ValueError("The number of original points and new points must be the same.")

        n = len(original_points)

        # Calculate average points
        avg_original = np.mean([p['position'] for p in original_points], axis=0)
        avg_new = np.mean([p['position'] for p in new_points], axis=0)

        # Calculate the average vector
        avg_vector = avg_new - avg_original

        # Find pairs whose vectors do not intersect the average vector
        non_intersecting_pairs = []
        for i, j in product(range(n), range(n)):
                orig_pos = original_points[i]['position'][:2]  # Use only x and y for 2D intersection check
                new_pos = new_points[j]['position'][:2]
                if not self.vector_intersects(avg_original[:2], avg_new[:2], orig_pos, new_pos):
                    vector = new_points[j]['position'] - original_points[i]['position']
                    distance = np.linalg.norm(vector)
                    non_intersecting_pairs.append((i, j, distance))

        # Sort non-intersecting pairs by distance
        non_intersecting_pairs.sort(key=lambda x: x[2])

        # Select the middle 4 pairs
        num_pairs = len(non_intersecting_pairs)
        print(f"Number of non-intersecting pairs: {num_pairs}")
        start_index = (num_pairs - 4) // 2
        selected_pairs = non_intersecting_pairs[start_index:start_index+4]

        return [(pair[0], pair[1]) for pair in selected_pairs]

    def heuristic(self, a, b):
        """Calculate the heuristic distance between two points."""
        return np.linalg.norm(a['position'] - b['position'])

    def find_surface_path(self, start_point, end_point, neighbor_dict):
        """Find the shortest path between two points on the surface using A* algorithm."""
        start_id = start_point['id']
        end_id = end_point['id']

        g_score = {start_id: 0}
        f_score = {start_id: self.heuristic(start_point, end_point)}
        came_from = {}
        open_set = [(f_score[start_id], start_id)]
        closed_set = set()

        while open_set:
            current_id = heapq.heappop(open_set)[1]

            if current_id == end_id:
                path = []
                while current_id in came_from:
                    path.append(current_id)
                    current_id = came_from[current_id]
                path.append(start_id)
                return path[::-1]

            closed_set.add(current_id)

            for neighbor_id in neighbor_dict[current_id]:
                if neighbor_id in closed_set:
                    continue

                tentative_g_score = g_score[current_id] + self.heuristic(self.surface_points[current_id], self.surface_points[neighbor_id])

                if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = g_score[neighbor_id] + self.heuristic(self.surface_points[neighbor_id], end_point)
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

        return None  # No path found

    def sphere_intersects_edge(self, point, radius, edge_tree):
        """Check if a sphere around a point intersects any edge points."""
        nearby_indices = edge_tree.query_ball_point(point, radius)
        return len(nearby_indices) > 0

    def spheres_intersect(self, p1, p2, radius):
        """Check if two spheres intersect."""
        return np.linalg.norm(p1 - p2) < 2 * radius

    def discretize_path(self, path, edge_tree, start_point, end_point):
        """
        Create a discrete path with steps approximately 2 * clearance_radius apart,
        ensuring spheres don't intersect each other or edge points.
        Always includes start and end points.
        """
        if len(path) < 2:
            return path  # Return the path as is if it has less than 2 points

        discrete_path = [path[0]]  # Start with the first point
        cumulative_distance = 0
        target_distance = 2 * self.clearance_radius

        for i in range(1, len(path) - 1):  # Exclude the last point for now
            current_point = self.surface_points[path[i]]['position']
            previous_point = self.surface_points[path[i-1]]['position']
            distance = np.linalg.norm(current_point - previous_point)
            cumulative_distance += distance

            if cumulative_distance >= target_distance:
                # Check if the sphere around this point intersects with any previous spheres or edge points
                if not any(self.spheres_intersect(current_point, self.surface_points[p]['position'], self.clearance_radius) for p in discrete_path) and \
                   not self.sphere_intersects_edge(current_point, self.clearance_radius, edge_tree):
                    discrete_path.append(path[i])
                    cumulative_distance = 0
                # If there's an intersection, we'll continue to the next point

        # Try to add points backwards from the end
        for i in range(len(path) - 2, 0, -1):
            if path[i] not in discrete_path:
                current_point = self.surface_points[path[i]]['position']
                if not any(self.spheres_intersect(current_point, self.surface_points[p]['position'], self.clearance_radius) for p in discrete_path) and \
                   not self.sphere_intersects_edge(current_point, self.clearance_radius, edge_tree):
                    discrete_path.append(path[i])

        # Always add the end point as the last step
        if path[-1] not in discrete_path:
            discrete_path.append(path[-1])

        # Ensure the first point is the start point
        discrete_path[0] = start_point['id']

        # Sort the discrete path to maintain the original order
        discrete_path.sort(key=lambda x: path.index(x))

        # Remove any remaining intersections
        i = 1
        while i < len(discrete_path) - 1:
            current_point = self.surface_points[discrete_path[i]]['position']
            if any(self.spheres_intersect(current_point, self.surface_points[discrete_path[j]]['position'], self.clearance_radius) 
                   for j in range(len(discrete_path)) if j != i):
                del discrete_path[i]
            else:
                i += 1

        return discrete_path

    def find_fastest_paths(self, original_points, new_points):
        """Find the fastest paths between original and new points."""
        neighbor_dict = {point['id']: point['neighbors'] for point in self.surface_points}
        
        # Create a KD-tree for edge points
        edge_points = [p['position'] for p in self.surface_points if p['type'] == 'edge']
        edge_tree = cKDTree(edge_points)

        pairs = self.find_matching_pairs(original_points, new_points)
        print(f"Number of matching pairs: {len(pairs)}")

        paths = []
        for orig_idx, new_idx in pairs:
            start_point = original_points[orig_idx]
            end_point = new_points[new_idx]
            path = self.find_surface_path(start_point, end_point, neighbor_dict)
            if path:
                discrete_path = self.discretize_path(path, edge_tree, start_point, end_point)
                paths.append((orig_idx, new_idx, discrete_path))
                print(f"Path found from original point {orig_idx} to new point {new_idx}: "
                      f"Original length: {len(path)}, Discretized length: {len(discrete_path)}")
            else:
                print(f"No path found from original point {orig_idx} to new point {new_idx}")

        return paths

    def generate_arc_points(self, start, end, num_points, start_normal, end_normal):
        """
        Generate points along an arc between start and end points,
        with the arc direction determined by the average of start and end normals.
        The height of the arc is controlled by self.arc_height_factor.
        
        Parameters:
        start (np.array): Start point of the arc
        end (np.array): End point of the arc
        num_points (int): Number of points to generate along the arc
        start_normal (np.array): Normal vector at the start point
        end_normal (np.array): Normal vector at the end point
        
        Returns:
        list: List of dictionaries containing arc point data
        """
        midpoint = (start + end) / 2
        vector = end - start
        
        # Use the average of start and end normals as the arc direction
        arc_direction = (start_normal + end_normal) / 2
        arc_direction = arc_direction / np.linalg.norm(arc_direction)
        
        # Ensure the arc_direction is perpendicular to the vector
        arc_direction = arc_direction - np.dot(arc_direction, vector) * vector / np.dot(vector, vector)
        arc_direction = arc_direction / np.linalg.norm(arc_direction)
        
        arc_points = []
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)
            # Use reversed angle to go from start to end
            angle = np.pi * (1 - t)
            
            # Apply arc height factor to the sin component
            # This directly controls how high the arc rises above the midpoint
            point = midpoint + 0.5 * vector * np.cos(angle) + \
                    0.5 * np.linalg.norm(vector) * self.arc_height_factor * arc_direction * np.sin(angle)
                    
            arc_points.append({
                'position': point,
                'type': 'arc',
                'normal': start_normal * (1 - t) + end_normal * t  # Interpolate normal
            })
        
        return arc_points

    def add_arc_points_to_path(self, path):
        """
        Add arc points to the existing path.
        
        Parameters:
        path (list): List of point indices representing the path
        
        Returns:
        list: New path with arc points included
        """
        new_path = []
        
        for i in range(len(path) - 1):
            start = self.surface_points[path[i]]
            end = self.surface_points[path[i+1]]
            
            new_path.append({
                'position': start['position'],
                'normal': start['normal'],
                'type': 'original',
                'id': start['id']
            })
            
            arc_points = self.generate_arc_points(
                start['position'], 
                end['position'], 
                self.num_arc_points, 
                start['normal'], 
                end['normal']
            )
            
            for j, point in enumerate(arc_points):
                point['id'] = f'arc_{i}_{j}'
                new_path.append(point)
        
        # Add the last point
        new_path.append({
            'position': self.surface_points[path[-1]]['position'],
            'normal': self.surface_points[path[-1]]['normal'],
            'type': 'original',
            'id': self.surface_points[path[-1]]['id']
        })
        
        return new_path

    def process_paths(self, paths):
        """
        Process all paths to include arc points.
        
        Parameters:
        paths (list): List of tuples (orig_idx, new_idx, path)
        
        Returns:
        list: List of processed paths with arc points included
        """
        processed_paths = []
        for orig_idx, new_idx, path in paths:
            new_path = self.add_arc_points_to_path(path)
            processed_paths.append(new_path)
        return processed_paths

    def order_paths_by_distance(self, paths):
        """
        Order paths by distance from start to end point.
        
        Parameters:
        paths (list): List of paths to order
        
        Returns:
        list: Ordered list of paths
        """
        path_distances = [np.linalg.norm(path[0]['position'] - path[-1]['position']) 
                         for path in paths]
        ordered_indices = np.argsort(path_distances)
        
        # Create ordered list of paths
        ordered_paths = [paths[i] for i in ordered_indices]
        
        return ordered_paths

    def plot_prism_surface_points(self, sample_size=None):
        """
        Plot the surface points of the prism with both original and rotated vectors, and the fastest paths.
        """
        if not all([self.original_result, self.rotated_result, self.processed_paths]):
            raise ValueError("Run find_trajectory method before plotting")
            
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Sample points if sample_size is provided
        plot_points = self.surface_points
        if sample_size is not None:
            plot_points = np.random.choice(self.surface_points, sample_size, replace=False)

        # Separate face and edge points
        positions = np.array([point['position'] for point in plot_points])
        point_types = np.array([point['type'] for point in plot_points])
        face_mask = point_types == 'face'
        edge_mask = point_types == 'edge'

        # Plot face and edge points
        ax.scatter(positions[face_mask, 0], positions[face_mask, 1], positions[face_mask, 2], 
                   c='blue', s=10, alpha=0.1, label='Face Points')
        ax.scatter(positions[edge_mask, 0], positions[edge_mask, 1], positions[edge_mask, 2], 
                   c='orange', s=10, alpha=0.1, label='Edge Points')

        print(f"Number of face points: {np.sum(face_mask)}")
        print(f"Number of edge points: {np.sum(edge_mask)}")

        center = self.original_result['center']

        # Plot original vector
        original_vector = np.array([0, 0, 1])
        original_intersection = self.original_result['intersection_point']
        ax.quiver(center[0], center[1], center[2],
                  original_vector[0], original_vector[1], original_vector[2],
                  length=np.linalg.norm(original_intersection - center),
                  normalize=False, color='green', linewidth=2, label='Original Vector')
        ax.scatter(original_intersection[0], original_intersection[1], original_intersection[2],
                   c='green', s=100, marker='*', label='Original Intersection')

        # Plot rotated vector
        rotated_vector = self.rotated_result['rotated_vector']
        rotated_intersection = self.rotated_result['intersection_point']
        ax.quiver(center[0], center[1], center[2],
                  rotated_vector[0], rotated_vector[1], rotated_vector[2],
                  length=np.linalg.norm(rotated_intersection - center),
                  normalize=False, color='red', linewidth=2, label='Rotated Vector')
        ax.scatter(rotated_intersection[0], rotated_intersection[1], rotated_intersection[2],
                   c='red', s=100, marker='*', label='Rotated Intersection')

        # Plot the fastest paths
        path_colors = ['cyan', 'magenta', 'yellow', 'black', 'purple']
        for i, path in enumerate(self.processed_paths):
            if i < len(path_colors):
                path_positions = np.array([p['position'] for p in path])
                ax.plot(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2], 
                        c=path_colors[i], linewidth=2, label=f'Path {i+1}')
                
                # Plot each step with appropriate marker
                for point in path:
                    position = point['position']
                    if point['type'] == 'original':
                        ax.scatter(position[0], position[1], position[2], c=path_colors[i], s=50, marker='o')
                        
                        # Plot clearance sphere for original points
                        u = np.linspace(0, 2 * np.pi, 20)
                        v = np.linspace(0, np.pi, 20)
                        x = position[0] + self.clearance_radius * np.outer(np.cos(u), np.sin(v))
                        y = position[1] + self.clearance_radius * np.outer(np.sin(u), np.sin(v))
                        z = position[2] + self.clearance_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                        ax.plot_surface(x, y, z, color=path_colors[i], alpha=0.1)
                    else:  # arc point
                        ax.scatter(position[0], position[1], position[2], c=path_colors[i], s=20, marker='s')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Prism Surface Points with Vectors and Fastest Paths')

        # Set axis limits
        max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                              positions[:, 1].max() - positions[:, 1].min(),
                              positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    def find_trajectory(self, centroid, initial_orientation, target_orientation):
        """
        Find trajectories from initial to target orientation.
        
        Parameters:
        centroid (np.array): Center position of the prism
        initial_orientation (np.array): Initial orientation in radians [rx, ry, rz]
        target_orientation (np.array): Target orientation in radians [rx, ry, rz]
        
        Returns:
        dict: Results containing original points, target points, and paths
        """
        # Create surface points if they don't exist
        if self.surface_points is None:
            self.create_prism_surface_points(centroid, initial_orientation)
        
        # Find surface points for original and target orientations
        self.original_result = self.find_surface_points(initial_orientation)
        self.rotated_result = self.find_surface_points(target_orientation)
        
        # Find and process paths
        paths = self.find_fastest_paths(
            self.original_result['closest_points'],
            self.rotated_result['closest_points']
        )
        self.processed_paths = self.process_paths(paths)
        
        # Order paths by distance
        self.processed_paths = self.order_paths_by_distance(self.processed_paths)
        
        return self.processed_paths,self.original_result
if __name__ == "__main__":    
        # Example usage
    # Create a pathfinder with a higher arc for easier visual tracking
    pathfinder = PrismSurfacePathfinder(
        dimensions=np.array([10, 5, 5]),
        resolution=0.1,
        clearance_radius=0.1,
        num_arc_points=4,
        min_intersection_distance=0.5,
        arc_height_factor=1.5  # Arcs will be 50% taller than standard semicircles
    )


    initial_centroid = np.array([0, 0, 0])
    initial_orientation = np.array([0, 0, 0])
    target_orientation = np.array([np.pi/3, np.pi/2, 0])

    # Find trajectories
    paths,initial_point = pathfinder.find_trajectory(initial_centroid, initial_orientation, target_orientation)

    pathfinder.plot_prism_surface_points()