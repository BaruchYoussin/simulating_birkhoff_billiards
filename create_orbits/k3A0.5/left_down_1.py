import general_lib
import billiard_Birkhoff_lib

k = 3
A = "0.5"
initial_phi, initial_p = general_lib.unpickle("starting_points_for_orbits_to_the_left.pkl")[1, :].tolist()
precision_main, precision_val = 200, 230
num_iter = 50 * 1000 * 1000
assert num_iter == 5e7
block_size = 1000 * 1000
assert num_iter == 50 * block_size
rectangle = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408", "0.906694307215364569412"]
# -- These are the boundary of the plot, "new_starting_points_above_below_approaching13.png"
# on which this starting point appears.
n_decimals_in_name_phi = 21
n_decimals_in_name_p = 24

orbit_filepath = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
    k, A, precision_main, precision_val, initial_phi, initial_p, num_iter, block_size, rectangle,
    n_decimals_in_name_phi, "../../", n_decimals_in_name_p)
print(f"Orbit created: {orbit_filepath}")
