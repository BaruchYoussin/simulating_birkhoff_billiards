# The two orbits from 2022 which are closest to the periodic point (one shows as the separatrices)
# and the starting points of the new orbit which are above and below the periodic point.

# based on billiards/python/billiardsDec2021/plot_k3Ahalf_approaching13.py
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
    ("../../../../billiardsDec2021/orbits_k3A0.5/"  # 8th
     "orbit_k3A0.5phi1.0471975511965977462p0.90669430721536456941prec200n_iter1000000.pkl",
     "black"),
    ("../../../../billiardsDec2021/orbits_k3A0.5/singular_orbit_prec100.pkl", "red", {"markersize": 5})  # periodic
]
mp.mp.dps = 50

phi_bounds_dict = {"xlow": "1.047197551196597745", "xhigh": "1.047197551196597747"}
p_bounds_dict = {"ylow": "0.906694307215364569408", "yhigh": "0.906694307215364569412"}

plotting_data = [plotting_lib.get_data_and_color(*filename_and_color) for filename_and_color
                       in filenames_and_colors]
plotting_lib.mpf_plot(ax, plotting_data, **phi_bounds_dict, **p_bounds_dict,
                      xticks_num=6, yticks_num=5, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)

phi_bounds_values = [mp.mpf(string_bound) for string_bound in phi_bounds_dict.values()]
p_bounds_values = [mp.mpf(string_bound) for string_bound in p_bounds_dict.values()]

# The initial point of the black orbit (the second one):
initial_point_black = (mp.pi / 3, mp.mpf("0.90669430721536456941"))

# Find the points on the orange curve on the same height as the initial point of the black curve:
orange_phi, orange_p, _ = plotting_data[0]
indexes,  = np.nonzero(np.abs(orange_p - initial_point_black[1]) < (p_bounds_values[1] - p_bounds_values[0]) * 2e-5)
orange_point_right, orange_point_left = zip(orange_phi[indexes].tolist(), orange_p[indexes].tolist())

# The orange curve on our plot is very close to separatrices.
# Find the approximate directions of the separatrices from it, to be used in another :
# The periodic point explicitly, from its calculation:
periodic = (mp.mpf(
    "1.047197551196597746154214461093167628065723133125035273658314864102605468762069666209344941780705642483"),
            mp.mpf(
    "0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015679"))
separatrix_vector_right = np.array(orange_point_right) - np.array(periodic)
separatrix_vector_left = np.array(orange_point_left) - np.array(periodic)
separatrix_slope_right = float(separatrix_vector_right[1] / separatrix_vector_right[0])
separatrix_slope_left = float(separatrix_vector_left[1] / separatrix_vector_left[0])
print(f"separatrix slopes: {separatrix_slope_right}, {separatrix_slope_left}")
# separatrix slopes: 0.0009300375438045088, -0.000930037510764236
# They are practically equal, so the plots are practically symmetric with respect to the vertical line
# passing through the periodic point.

# The new initial points between the black curve and the orange one:
new_top_left = tuple((np.array(initial_point_black) + np.array(orange_point_left)) / 2)
new_top_right = tuple((np.array(initial_point_black) + np.array(orange_point_right)) / 2)
# The new initial points below the periodic:
bottom_p = 2 * periodic[1] - initial_point_black[1]
new_bottom_left = (new_top_left[0], bottom_p)
new_bottom_right = (new_top_right[0], bottom_p)

# Shift these points towards the separatrices (down and up) by different values:
vertical_distance = initial_point_black[1] - periodic[1]
new_top_left = (new_top_left[0], new_top_left[1] - 0.1 * vertical_distance)
new_bottom_left = (new_bottom_left[0], new_bottom_left[1] + 0.2 * vertical_distance)
new_bottom_right = (new_bottom_right[0], new_bottom_right[1] + 0.3 * vertical_distance)

new_point_strings = [[mp.nstr(coord, mp.mp.dps) for coord in point]
                     for point in (new_top_left, new_top_right, new_bottom_left, new_bottom_right)]
print("The new initial points:")
for point in new_point_strings:
    print(point)
# The new initial points:
# ['1.0471975511965977459651997100036728260448414392093', '0.90669430721536456940996481758960653575029143297692']
# ['1.0471975511965977463432832550525267124396437207693', '0.90669430721536456941000003006091532668894479785147']
# ['1.0471975511965977459651997100036728260448414392093', '0.90669430721536456940936708034301662102564491748964']
# ['1.0471975511965977463432832550525267124396437207693', '0.90669430721536456940940224254618236429440727727194']

# Plot the initial point of the black orbit, points on the orange curve and the new initial points:
points = [initial_point_black, orange_point_left, orange_point_right, new_top_left, new_top_right, new_bottom_left,
          new_bottom_right]
phi_values, p_values = (np.array(values) for values in zip(*points))
x_points = general_lib.rescale_array_to_float(phi_values, *phi_bounds_values)
y_points = general_lib.rescale_array_to_float(p_values, *p_bounds_values)
ax.scatter(x_points, y_points, s=5, c=["black"] + 2 * ["orange"] + ["blue", "green", "green", "blue"])
fig.savefig("../../plots/k3A0.5/new_starting_points_above_below_approaching13.png")

general_lib.pickle_one_object(np.array(new_point_strings),
                              "../../create_orbits/k3A0.5/starting_points_for_orbits_up_and_down.pkl")
