# This script creates an orbit with the starting point located above and to the left from
# the periodic hyperbolic point with phi = pi/3.
import billiard_Birkhoff_lib

k = 3
A = "0.5"
initial_phi, initial_p = ['1.0471975511965977459651997100036728260448414392093',
                          '0.90669430721536456940996481758960653575029143297692']
precision_main, precision_val = 120, 140
num_iter = 5 * 1000 * 1000
assert num_iter == 5e6
block_size = 200 * 1000
assert num_iter == 25 * block_size
rectangle = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408", "0.906694307215364569412"]
# --keep only the points within this rectangle; the first two values are the boundaries in phi,
# the second two are the boundaries in p.
n_decimals_in_name_phi = 21
n_decimals_in_name_p = 24

orbit_filepath = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
    k, A, precision_main, precision_val, initial_phi, initial_p, num_iter, block_size, rectangle,
    n_decimals_in_name_phi, "../../", n_decimals_in_name_p)
print(f"Orbit created: {orbit_filepath}")
# Max discrepancy: 4.24e-75
# Orbit created: ../../orbits/k3A0.5/orbit_k3A0.5phi1.04719755119659774597p0.906694307215364569409965prec120n_iter5000000/phi1.047197551196597745to1.047197551196597747p0.906694307215364569408to0.906694307215364569412/orbit.pkl
# Takes about 0.5-0.55 hours per block of 200,000 iterations (including validation), ~13 hours total.
