# This file plots the separatrices for the hyperbolic points of order 9 that was
# found in create_orbits/k3A0.5/periodic_orbit_k3A0.5.py; these separtrices were found with the precision of 50 decimals
# in the files separatrix_9_..._50_n.py , and also obtained by flipping them w.r. to phi = pi/3.

import matplotlib.pyplot as plt
import mpmath as mp

import plotting_lib

plt.rcParams.update({'figure.autolayout': True})

fig, ax = plt.subplots(dpi=1200, figsize=[6.4, 6.4], layout='constrained')

mp.mp.dps = 50

periodic_filename_and_color = ("../../orbits/k3A0.5/singular_orbit_prec100.pkl", "red", {"markersize": 5})
periodic_data_and_color = plotting_lib.get_data_and_color(*periodic_filename_and_color)
periodic_phi, periodic_p = (periodic_data_and_color[i][1] for i in range(2))

# forward separatrices, near their hyperbolic points:
separatrix_filenames_and_colors_and_kwargs_up_left = [
    ((f"../../orbits/k3A0.5/separatrices/9up_left50_{n}/separatrix.pkl", "black",
      {"label": "black: up left, comes back from up right"} if n == 0 else {}),
     {"front": 0}) for n in range(6)]
separatrix_filenames_and_colors_and_kwargs_down_left = [
    ((f"../../orbits/k3A0.5/separatrices/9down_left50_{n}/separatrix.pkl", "blue",
      {"label": "blue: down left, comes back from down right"} if n == 0 else {}),
     {"front": 0}) for n in range(6)]
# backward separatrices, near the hyperbolic point to which they "come back":
separatrix_filenames_and_colors_and_kwargs_up_right = [
    ((f"../../orbits/k3A0.5/separatrices/9up_left50_{n}/separatrix.pkl", "cyan",
      {"label": "cyan: up right, comes back from up left"} if n == 0 else {}),
     {"back": 0, "flip_phi": mp.pi/3, "period_phi": 2 * mp.pi}) for n in range(6)]
separatrix_filenames_and_colors_and_kwargs_down_right = [
    ((f"../../orbits/k3A0.5/separatrices/9down_left50_{n}/separatrix.pkl", "orange",
      {"label": "orange: down right, comes back from down left"} if n == 0 else {}),
     {"back": 0, "flip_phi": mp.pi/3, "period_phi": 2 * mp.pi}) for n in range(6)]

separatrix_filenames_and_colors_and_kwargs_up = (separatrix_filenames_and_colors_and_kwargs_up_right
                                                 + separatrix_filenames_and_colors_and_kwargs_up_left)
separatrix_filenames_and_colors_and_kwargs_down = (separatrix_filenames_and_colors_and_kwargs_down_right
                                                   + separatrix_filenames_and_colors_and_kwargs_down_left)

# up left:
plotting_data = [periodic_data_and_color] + [
    plotting_lib.get_separatrix_data_and_color(*filename_and_color, **kwargs)
    for filename_and_color, kwargs in separatrix_filenames_and_colors_and_kwargs_up]
rectangle_to_plot = [  # same as in separatrices_9_100decimals.py
    mp.mpf("1.047197551196597746154214461093167628065723133"),
    mp.mpf("1.0471975511965977461542144610931676280657231332"),
    mp.mpf("0.906694307215364569409648377968342567234300627074"),
    mp.mpf("0.906694307215364569409648377968342567234300627075")]
plotting_lib.mpf_plot_based(ax, plotting_data, *rectangle_to_plot, periodic_phi, periodic_p,
                            xtick=1e-47, ytick=1e-49, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
fig.legend(loc="outside lower center",
           title=f"The hyperbolic point is at\nphi = {mp.nstr(periodic_phi, 49)},\np = {mp.nstr(periodic_p, 50)}.\n"
                 f"The ticks on the axes are offsets from its coordinates.\n"
                 f"This plot shows only the separatrices in the up left direction.\n"
                 f"The directions of separatrices from the hyperbolic point, by color, are as follows:")
fig.savefig("../../plots/k3A0.5/separatrices_9_50dec_up_left.png")

fig, ax = plt.subplots(dpi=1200, figsize=[6.4, 6.4], layout='constrained')

# down left:
plotting_data = [periodic_data_and_color] + [
    plotting_lib.get_separatrix_data_and_color(*filename_and_color, **kwargs)
    for filename_and_color, kwargs in separatrix_filenames_and_colors_and_kwargs_down]
rectangle_to_plot = [  # same as in separatrices_9_100decimals.py
    mp.mpf("1.0471975511965977461542144610931676280657231322"),
    mp.mpf("1.047197551196597746154214461093167628065723134"),
    mp.mpf("0.906694307215364569409648377968342567234300627072"),
    mp.mpf("0.906694307215364569409648377968342567234300627078")]
plotting_lib.mpf_plot_based(ax, plotting_data, *rectangle_to_plot, periodic_phi, periodic_p,
                            xtick=1e-46, ytick=1e-48, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
fig.legend(loc="outside lower center",
           title=f"The hyperbolic point is at\nphi = {mp.nstr(periodic_phi, 49)},\np = {mp.nstr(periodic_p, 50)}.\n"
                 f"The ticks on the axes are offsets from its coordinates.\n"
                 f"This plot shows only the separatrices in the down left direction.\n"
                 f"The directions of separatrices from the hyperbolic point, by color, are as follows:")
fig.savefig("../../plots/k3A0.5/separatrices_9_50dec_down_left.png")

fig, ax = plt.subplots(dpi=1200, figsize=[6.4, 6.4], layout='constrained')

# enlarge the plot somewhat away from the hyperbolic point in a place where the coming back separatrix
# already diverges from the going out, but a little:
rectangle_to_plot = [  # same as on the plot with all directions
    mp.mpf("1.0471975511965977461542144610931676280657231322"),
    mp.mpf("1.0471975511965977461542144610931676280657231326"),
    mp.mpf("0.906694307215364569409648377968342567234300627073"),
    mp.mpf("0.906694307215364569409648377968342567234300627075")]
plotting_lib.mpf_plot_based(ax, plotting_data, *rectangle_to_plot, periodic_phi, periodic_p,
                            xtick=2e-47, ytick=1e-49, rotate_xticklabels=True)
ax.set_xlabel("phi", loc="right")
ax.set_ylabel("p", loc="top", rotation=0)
fig.legend(loc="outside lower center",
           title=f"The hyperbolic point is at\nphi = {mp.nstr(periodic_phi, 49)},\np = {mp.nstr(periodic_p, 50)}.\n"
                 f"The ticks on the axes are offsets from its coordinates.\n"
                 f"This plot shows only the separatrices in the down left direction.\n"
                 f"The directions of separatrices from the hyperbolic point, by color, are as follows:")
fig.savefig("../../plots/k3A0.5/separatrices_9_50dec_down_left_enlarged_away.png")

fig, ax = plt.subplots(dpi=1200, figsize=[6.4, 6.4], layout='constrained')
