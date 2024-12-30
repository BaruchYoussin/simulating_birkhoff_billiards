# The orbit from 2022 which is closest to the periodic point, the separatrices
# and the starting points of the new orbit which are to the left of the periodic point.

# based on billiards/python/billiardsDec2021/plot_k3Ahalf_approaching11.py
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import general_lib
import plotting_lib

plt.rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots(dpi=1200)

filenames_and_colors = [
    ("../../../../billiardsDec2021/orbits_k3A0.5/"  # 6th
     "orbit_k3A0.5phi1.0471975511965977461p0.90669430721536456941prec200n_iter1000000.pkl",
     "orange"),
    ("../../../../billiardsDec2021/orbits_k3A0.5/singular_orbit_prec100.pkl", "red", {"markersize": 5})  # periodic
]
mp.mp.dps = 50

phi_bounds_dict = {"xlow": "1.04719755119659774614", "xhigh": "1.04719755119659774617"}
p_bounds_dict = {"ylow": "0.90669430721536456940964", "yhigh": "0.90669430721536456940966"}

plotting_data = [plotting_lib.get_data_and_color(*filename_and_color) for filename_and_color
                       in filenames_and_colors]
plotting_lib.mpf_plot(ax, plotting_data, **phi_bounds_dict, **p_bounds_dict,
                      xticks_num=6, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)

phi_bounds_values = [mp.mpf(string_bound) for string_bound in phi_bounds_dict.values()]
p_bounds_values = [mp.mpf(string_bound) for string_bound in p_bounds_dict.values()]

# The periodic point explicitly, from its calculation:
periodic = (mp.mpf(
    "1.047197551196597746154214461093167628065723133125035273658314864102605468762069666209344941780705642483"),
            mp.mpf(
    "0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015679"))
# The approximate directions of the separatrix slopes, as found in old_orbits_and_new_starting_points1.py :
separatrix_slope = 0.000930037525
# Plot the separatrices in black:
separatrix_point0 = (phi_bounds_values[1], periodic[1] + (phi_bounds_values[1] - periodic[0]) * separatrix_slope)
separatrix_point1 = (phi_bounds_values[1], periodic[1] - (phi_bounds_values[1] - periodic[0]) * separatrix_slope)
separatrix_coords_phi, separatrix_coords_p = tuple(zip(periodic, separatrix_point0, separatrix_point1))
rescaled_separatrix_points = list(zip(general_lib.rescale_array(np.array(separatrix_coords_phi),
                                                                *phi_bounds_values).astype(float).tolist(),
                                      general_lib.rescale_array(np.array(separatrix_coords_p),
                                                                *p_bounds_values).astype(float).tolist()))
plt.axline(*rescaled_separatrix_points[:2], c="black")
plt.axline(rescaled_separatrix_points[0], rescaled_separatrix_points[2], c="black")

# Add the new initial points:
new_left_up_strings = ("1.04719755119659774615", "0.9066943072153645694096505")
new_left_down_strings = ("1.0471975511965977461505", "0.9066943072153645694096465")
new_left_up = (mp.mpf(string) for string in new_left_up_strings)
new_left_down = (mp.mpf(string) for string in new_left_down_strings)

phi_values, p_values = (np.array(values) for values in zip(new_left_up, new_left_down))
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=["blue", "brown"])

fig.savefig("../../plots/k3A0.5/new_starting_points_to_the_left_approaching11.png")

general_lib.pickle_one_object(np.array([new_left_up_strings, new_left_down_strings]),
                              "../../create_orbits/k3A0.5/starting_points_for_orbits_to_the_left.pkl")
