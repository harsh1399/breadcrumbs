import itertools
from numba import jit,cuda
import numpy as np
def volume_of_tetrahedron(p1, p2, p3, p4):
    # Vectors from p1 to p2, p3, and p4
    AB = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    AC = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    AD = (p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2])

    # Direct calculation of the cross product components
    cross_product_x = AB[1] * AC[2] - AB[2] * AC[1]
    cross_product_y = AB[2] * AC[0] - AB[0] * AC[2]
    cross_product_z = AB[0] * AC[1] - AB[1] * AC[0]

    # Dot product of AD with the cross product of AB and AC
    scalar_triple_product = (
        AD[0] * cross_product_x +
        AD[1] * cross_product_y +
        AD[2] * cross_product_z
    )

    # The volume of the tetrahedron
    volume = abs(scalar_triple_product) / 6.0
    return volume

def read_points(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z, n = line.strip('()\n').split(', ')
            points.append((float(x), float(y), float(z), int(n)))
    return points
# @jit(target_backend='cuda')
def find_smallest_tetrahedron(points):
    min_volume = float('inf')
    best_combination = None
    num_combinations_checked = 0

    for comb in itertools.combinations(enumerate(points), 4):
        indices, pts = zip(*comb)
        if np.sum(points[indices,3]) == 100:
            volume = volume_of_tetrahedron(*pts)
            if volume < min_volume:
                min_volume = volume
                best_combination = indices
            num_combinations_checked += 1

            # Print debug information
            if num_combinations_checked % 1000 == 0:
                print(f"Checked {num_combinations_checked} combinations, current min volume: {min_volume}")

    return sorted(best_combination) if best_combination else None

# Read points from files
points_small = np.array(read_points('points/points_small.txt'))
points_large = np.array(read_points('points/points_large.txt'))

# # Debugging statements to check file reading
# print(f"Small points read: {len(points_small)}")
# print(f"Large points read: {len(points_large)}")

# Find the smallest tetrahedron
small_result = find_smallest_tetrahedron(points_small)
large_result = find_smallest_tetrahedron(points_large)

print(f"Small file tetrahedron indices: {small_result}")
print(f"Large file tetrahedron indices: {large_result}")
