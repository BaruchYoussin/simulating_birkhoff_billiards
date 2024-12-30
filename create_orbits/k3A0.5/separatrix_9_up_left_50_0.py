# This file creates the separatrix for the 9-periodic point at phi = p/3 found in
# create_orbits/k3A0.5/periodic_orbit_k3A0.5.py, in the upward left direction.
# Precision of the separatrix: 50 decimals achieved by working precision 110 decimals (validation: 130).
# Based on separatrix_3_down_right.py and separatrix_9_down_left_50_0.py
# This file is 0th (0-based) of the 6 files creating the separtrices for different starting points, so that together
# they yield a denser separatrix.

import mpmath as mp
import numpy as np
import general_lib
import billiard_Birkhoff_lib

k = 3
A = "0.5"
path_to_orbit_ambient = "../../"

dir_name = "9up_left50_0"

period = 9
precision_main, precision_val, precision = 110, 130, 10 ** (-50)
hyperbolic_point = general_lib.unpickle("../../orbits/k3A0.5/singular_orbit_prec100.pkl")[1, 1:]  # at phi = pi/3

approach_separatrix_direction = np.array([-1, 0.00093])  # the eigenvalue is 1.147, see periodic_orbit_k3A0.5.py
other_separatrix_direction = np.array([-1, -0.00093])  # the eigenvalue is 0.87
forward = False
iteration_limit = 10000
mp.mp.dps = precision_val  # for calculating the starting points
approaching_ratio_between_files = 0.99953  # for 300 iterations between 6 files, 50 iterations per file:
# 0.99953 ** (-300) = 1.1514628130315907, which is somewhat more than the eigenvalue, ~ 1.147.
starting_point1_all_files = hyperbolic_point + 1e-6 * approach_separatrix_direction + 1e-8 * other_separatrix_direction
starting_point2_all_files = hyperbolic_point + 1e-6 * approach_separatrix_direction - 1e-8 * other_separatrix_direction
starting_point_pairs = list(general_lib.InitialPointsForFindingSeparatrix(
    hyperbolic_point, approaching_ratio_between_files).starting_points(
    starting_point1_all_files, starting_point2_all_files, 6))

file_number = 0

starting_point1, starting_point2 = starting_point_pairs[file_number]
separating_coefficient = 1
approaching_ratio = approaching_ratio_between_files ** 6
num_new_orbits = 50
block_size = 1
num_steps_back = 8300  # 900 + 5800 additional in the original runs for 40 decimals + 1600 steps back needed to get
# from the distance of 1e-40 to 1e-50 (log(1e60)/log(1.147) ~ 168 * 9 ~ 1512)
separatrix_path = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.separatrix_by_blocks(
    k, A, precision_main, precision_val, period, iteration_limit, hyperbolic_point,
    approach_separatrix_direction, other_separatrix_direction, precision, separating_coefficient, forward,
    starting_point1, starting_point2, approaching_ratio, num_new_orbits, block_size, num_steps_back,
    path_to_orbit_ambient, dir_name, verbose=True)
# Takes up to 7.5 hours per orbit
# Max discrepancy: 2.5741464404882526e-58
print(f"Completed! separatrix_path: {separatrix_path}")
# Completed! separatrix_path: ../../orbits/k3A0.5/separatrices/9up_left50_0/separatrix.pkl
