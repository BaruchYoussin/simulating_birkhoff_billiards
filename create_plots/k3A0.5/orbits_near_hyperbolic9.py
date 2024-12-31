# This script plots the orbits near hyperbolic periodic point of order 9, that were created by the scripts
# above_left_1.py and left_down_1.py .

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import general_lib
import plotting_lib

mp.mp.dps = 40

plt.rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots(dpi=1200)

filenames_and_colors = [
    ("../../orbits/k3A0.5/orbit_k3A0.5phi1.04719755119659774597p0.906694307215364569409965prec120n_iter5000000/"
     "phi1.047197551196597745to1.047197551196597747p0.906694307215364569408to0.906694307215364569412/orbit.pkl",
     "blue"),
    ("../../orbits/k3A0.5/orbit_k3A0.5phi1.04719755119659774615p0.906694307215364569409646prec120n_iter5000000/"
     "phi1.047197551196597745to1.047197551196597747p0.906694307215364569408to0.906694307215364569412/orbit.pkl",
     "brown"),
    ("../../orbits/k3A0.5/singular_orbit_prec100.pkl", "red", {"markersize": 5, "label": "hyperbolic periodic point"})]
plotting_data = [plotting_lib.get_data_and_color(*filename_and_color) for filename_and_color in filenames_and_colors]

# plot also the initial points of the orbits:
initial_points = [('1.0471975511965977459651997100036728260448414392093',
                   '0.90669430721536456940996481758960653575029143297692'),
                  ("1.0471975511965977461505", "0.9066943072153645694096465")]
phi_values, p_values = (np.array([mp.mpf(value) for value in values]) for values in zip(*initial_points))

rectangle = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408", "0.906694307215364569412"]

plotting_lib.mpf_plot(ax, plotting_data, *rectangle, xticks_num=5, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
rectangle_values = [mp.mpf(value) for value in rectangle]
phi_bounds_values, p_bounds_values = rectangle_values[:2], rectangle_values[2:]
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=["blue", "brown"])
ax.legend(loc="upper right")
fig.savefig("../../plots/k3A0.5/orbits1.png")

fig, ax = plt.subplots(dpi=1200)
rectangle = ["1.04719755119659774614", "1.04719755119659774617", "0.90669430721536456940964",
                     "0.90669430721536456940966"]
plotting_lib.mpf_plot(ax, plotting_data, *rectangle, xticks_num=4, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
rectangle_values = [mp.mpf(value) for value in rectangle]
phi_bounds_values, p_bounds_values = rectangle_values[:2], rectangle_values[2:]
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=["blue", "brown"])
ax.legend(loc="upper right")
fig.savefig("../../plots/k3A0.5/orbits2.png")
