# This file creates the plot of the 6 orbits whose starting points were defined and plotted in
# old_orbits_and_new_starting_points1.py and old_orbits_and_new_starting_points2.py, in the same rectangle
# as used in the first of these files.
# The orbits whose initial points were defined and plotted in the second of the above files, show as separatrices as
# they get too close to the periodic orbit for the resolution of the plot.
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

rectangle = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408", "0.906694307215364569412"]
n_decimals_in_name_phi = 21
n_decimals_in_name_p = 24
num_iter = 50 * 1000 * 1000
up_and_down_points = general_lib.unpickle("../../create_orbits/k3A0.5/starting_points_for_orbits_up_and_down.pkl").tolist()
left_points = general_lib.unpickle("../../create_orbits/k3A0.5/starting_points_for_orbits_to_the_left.pkl").tolist()
six_points = up_and_down_points + left_points
# Each orbit lies in its own directory, and has one of two possible standard names:
billiard_orbit = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 200)
mp.mp.dps = 50  # do not need 200 for the plot
dir_paths = ["../../"
             + billiard_orbit.dir_orbit_rectangle(init_phi, init_p, num_iter, rectangle,
                                                  n_decimals_in_name_phi, n_decimals_in_name_p)
             for [init_phi, init_p] in six_points]
filenames = [current_path if os.path.exists(current_path) else os.path.join(dir_path, "clipped_orbit_previous.pkl")
             for dir_path in dir_paths for current_path in [os.path.join(dir_path, "orbit.pkl")]]
colors = ["blue", "green", "green", "blue", "blue", "brown"]
filenames_and_colors = list(zip(filenames, colors))
filenames_and_colors = filenames_and_colors + [("../../orbits/k3A0.5/singular_orbit_prec100.pkl",  "red",
                                                {"markersize": 5})]
plotting_data = [plotting_lib.get_data_and_color(*filename_and_color) for filename_and_color in filenames_and_colors]
plotting_lib.mpf_plot(ax, plotting_data, *rectangle, xticks_num=5, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
# plot the initial points for the up and down orbits only:
phi_values, p_values = (np.array([mp.mpf(value) for value in values]) for values in zip(*up_and_down_points))
rectangle_values = [mp.mpf(value) for value in rectangle]
phi_bounds_values, p_bounds_values = rectangle_values[:2], rectangle_values[2:]
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=[colors_lib.get_plotting_color(color) for color in colors[:4]])
fig.savefig("../../plots/k3A0.5/first_6_orbits_large_rectangle.png")
