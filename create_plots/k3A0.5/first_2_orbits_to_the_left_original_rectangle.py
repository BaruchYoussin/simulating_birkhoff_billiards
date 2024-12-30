# This file creates the plot of the 2 orbits whose starting points were defined to the left of the periodic point
# and plotted in old_orbits_and_new_starting_points2.py, in the same rectangle as used in this file.
# This file plots the orbits also while they are still being calculated.
import os

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import billiard_Birkhoff_lib
import general_lib
import plotting_lib
import colors_lib

plt.rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots(dpi=1200)

rectangle_in_orbit_calculation = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408",
                                  "0.906694307215364569412"]
n_decimals_in_name_phi = 21
n_decimals_in_name_p = 24
num_iter = 50 * 1000 * 1000
points = general_lib.unpickle("../../create_orbits/k3A0.5/starting_points_for_orbits_to_the_left.pkl").tolist()
# Each orbit lies in its own directory, and has one of two possible standard names:
billiard_orbit = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 200)
mp.mp.dps = 50  # do not need 200 for the plot
dir_paths = ["../../"
             + billiard_orbit.dir_orbit_rectangle(init_phi, init_p, num_iter, rectangle_in_orbit_calculation,
                                                  n_decimals_in_name_phi, n_decimals_in_name_p)
             for [init_phi, init_p] in points]
filenames = [current_path if os.path.exists(current_path) else os.path.join(dir_path, "clipped_orbit_previous.pkl")
             for dir_path in dir_paths for current_path in [os.path.join(dir_path, "orbit.pkl")]]
colors = ["blue", "brown"]
filenames_and_colors = list(zip(filenames, colors))
filenames_and_colors = filenames_and_colors + [("../../orbits/k3A0.5/singular_orbit_prec100.pkl",  "red",
                                                {"markersize": 5})]
plotting_data = [plotting_lib.get_data_and_color(*filename_and_color) for filename_and_color in filenames_and_colors]
rectangle_to_plot = ["1.04719755119659774614", "1.04719755119659774617", "0.90669430721536456940964",
                     "0.90669430721536456940966"]
plotting_lib.mpf_plot(ax, plotting_data, *rectangle_to_plot, xticks_num=4, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
# plot the initial points for the up and down orbits only:
phi_values, p_values = (np.array([mp.mpf(value) for value in values]) for values in zip(*points))
rectangle_values = [mp.mpf(value) for value in rectangle_to_plot]
phi_bounds_values, p_bounds_values = rectangle_values[:2], rectangle_values[2:]
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=[colors_lib.get_plotting_color(color) for color in colors[:4]])
fig.savefig("../../plots/k3A0.5/first_2_orbits_to_the_left_original_rectangle.png")
