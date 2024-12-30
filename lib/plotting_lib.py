import numpy as np
import mpmath as mp
import matplotlib.axes
import general_lib
import colors_lib


def _to_mpf(value) -> mp.mpf:
    """Converts a value into mp.mpf; if it is not a string or an mpf, first converts into a string and then to mpf."""
    if not isinstance(value, mp.mpf):
        value = str(value)
    return mp.mpf(value)


def _get_ticks(ticks_start, ticks_end, ticks_num, low, high) -> tuple:
    """Create ticks for a plot obtained by rescaling.

    :param ticks_start: The first tick value, possibly a string for mpmath.
    :param ticks_end: The last tick value, possibly a string for mpmath.
    :param ticks_num: The total number of ticks including the start and the end.
    :param low, high: The low and the high values for rescaling the data, so that the rescaled data could be converted
        to float from mp.mpf.
    :returns scaled_ticks, ticks
        (scaled_ticks are compared with the scaled data while ticks is a list of strings which are displayed.
    """
    ticks_start = _to_mpf(ticks_start)
    ticks_end = _to_mpf(ticks_end)
    tick_values = general_lib.linspace(ticks_start, ticks_end, ticks_num)
    scaled_ticks = general_lib.rescale_array_to_float(np.array(tick_values), low, high).tolist()
    mp.pretty = True
    # The following code worked well in most cases but in some created tails 0.49999999.. instead of 0.5:
    # ticks = [str(tick) for tick in tick_values]
    # This was caused by numerical errors in dividing the interval from ticks_start to ticks_end.
    # Drop the last available digit to discard such errors:
    ticks = [mp.nstr(tick, mp.mp.dps - 1) for tick in tick_values]
    return scaled_ticks, ticks


def _get_based_ticks(low, high, base_value, tick_value) -> tuple:
    """Create ticks relative to a base value, for a plot obtained by rescaling.

    :param low, high: The low and the high values for rescaling the data, so that the rescaled data could be converted
        to float from mp.mpf.
    :param base_value: the base value; all the ticks are understood as offsets to this value.
    :param tick_value: the value of one tick.
    :returns scaled_ticks, ticks
        (scaled_ticks are compared with the scaled data while ticks is a list of strings which are displayed.
    """
    low = _to_mpf(low)
    high = _to_mpf(high)
    base_value = _to_mpf(base_value)
    tick_value = _to_mpf(tick_value)
    tick_values = general_lib.two_sided_range(low - base_value, high - base_value, tick_value)
    scaled_ticks = general_lib.rescale_array_to_float(np.array(tick_values) + base_value, low, high).tolist()
    mp.pretty = True
    # See the explanation for the following line in _get_ticks(.):
    ticks = [mp.nstr(tick, mp.mp.dps - 1) for tick in tick_values]
    return scaled_ticks, ticks


def mpf_plot(axes: matplotlib.axes.Axes, plotting_data: list, xlow, xhigh, ylow, yhigh, xticks_num: int,
             yticks_num: int, xticks_start=None, xticks_end=None, yticks_start=None, yticks_end=None,
             rotate_xticklabels: bool = False, x_period=None) -> None:
    """Plot the data after clipping, using mpmath to cover the case when clipping goes beyond machine precision.

    mp.mp.dps should be set by the caller to a value sufficient for the precision of xdata and ydata.
    :param axes: whether to plot.
    :param plotting_data: a list of tuples (xdata, ydata, color) or (xdata, ydata, color, kwargs)
        where xdata and ydata are array-like
        of equal length, color is one color to plot them, and kwargs, if present, are passed to axes.plot(..).
        The following default values override the default values of axes.plot(..):
        linestyle="", marker=".", markersize=1.
    :param xlow: the lower limit to clip on the x axis, possibly a string for mpmath.
    :param xhigh: the upper limit to clip on the x axis, possibly a string for mpmath.
    :param ylow: the lower limit to clip on the y axis, possibly a string for mpmath.
    :param yhigh: the upper limit to clip on the y axis, possibly a string for mpmath.
    :param xticks_num: The total number of xticks including the start and the end.
    :param yticks_num: The total number of yticks including the start and the end.
    :param xticks_start: The first value of xtick, possibly a string for mpmath; default: xlow.
    :param xticks_end: The last value of xtick, possibly a string for mpmath; default: xhigh.
    :param yticks_start: The first value of ytick, possibly a string for mpmath; default: ylow.
    :param yticks_end: The last value of ytick, possibly a string for mpmath; default: yhigh.
    :param rotate_xticklabels: if True, rotate x tick labels 45 degrees; use this when the labels are long.
    :param x_period: if not None, understood as the period in the x axis.
        All x values are changed by the multiples of x_period so that they belong in the interval [xlow, xlow + period).
        (In this way 0 may be between xlow and xhigh and all x values will be clipped into this interval correctly
        by subtracting x_period from them if necessary.)
    """
    xlow = _to_mpf(xlow)
    xhigh = _to_mpf(xhigh)
    ylow = _to_mpf(ylow)
    yhigh = _to_mpf(yhigh)
    if xticks_start is None:
        xticks_start = xlow
    if xticks_end is None:
        xticks_end = xhigh
    if yticks_start is None:
        yticks_start = ylow
    if yticks_end is None:
        yticks_end = yhigh
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    scaled_xticks, xticks = _get_ticks(xticks_start, xticks_end, xticks_num, xlow, xhigh)
    scaled_yticks, yticks = _get_ticks(yticks_start, yticks_end, yticks_num, ylow, yhigh)
    xticks_kwargs = {"rotation": 45, "horizontalalignment": 'right'} if rotate_xticklabels else {}
    axes.set_xticks(scaled_xticks, xticks, **xticks_kwargs)
    axes.set_yticks(scaled_yticks, yticks)
    for one_plot in plotting_data:
        assert len(one_plot) in [3, 4]
        if len(one_plot) == 3:
            xdata, ydata, color = one_plot
            kwargs = {}
        else:
            xdata, ydata, color, kwargs = one_plot
        clipped_x, clipped_y = general_lib.clip_pair_of_arrays(xdata, ydata, xlow, xhigh, ylow, yhigh, x_period)
        scaled_x = general_lib.rescale_array_to_float(clipped_x, xlow, xhigh)
        scaled_y = general_lib.rescale_array_to_float(clipped_y, ylow, yhigh)
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        if "markersize" not in kwargs:
            kwargs["markersize"] = 1
        axes.plot(scaled_x, scaled_y, color=color, **kwargs)


def mpf_plot_based(axes: matplotlib.axes.Axes, plotting_data: list, xlow, xhigh, ylow, yhigh, xbase, ybase, xtick,
             ytick, rotate_xticklabels: bool = False, x_period=None) -> None:
    """Plot the data after clipping; use mpmath; make ticks relative to a base point.

    This covers the case when clipping is to a rectangle so small that the absolute ticks would have too many decimals.
    mp.mp.dps should be set by the caller to a value sufficient for the precision of xdata and ydata.
    :param axes: whether to plot.
    :param plotting_data: a list of tuples (xdata, ydata, color) or (xdata, ydata, color, kwargs)
        where xdata and ydata are array-like
        of equal length, color is one color to plot them, and kwargs, if present, are passed to axes.plot(..).
        The following default values override the default values of axes.plot(..):
        linestyle="", marker=".", markersize=1.
    :param xlow: the lower limit to clip on the x axis, possibly a string for mpmath.
    :param xhigh: the upper limit to clip on the x axis, possibly a string for mpmath.
    :param ylow: the lower limit to clip on the y axis, possibly a string for mpmath.
    :param yhigh: the upper limit to clip on the y axis, possibly a string for mpmath.
    :param xbase: the base value for x, all x ticks are relative to this value.
    :param ybase: the base value for y, all y ticks are relative to this value.
    :param xtick: the value of one tick on the x axis.
    :param ytick: the value of one tick on the y axis.
    :param rotate_xticklabels: if True, rotate x tick labels 45 degrees; use this when the labels are long.
    :param x_period: if not None, understood as the period in the x axis.
        All x values are changed by the multiples of x_period so that they belong in the interval [xlow, xlow + period).
        (In this way 0 may be between xlow and xhigh and all x values will be clipped into this interval correctly
        by subtracting x_period from them if necessary.)
    """
    xlow = _to_mpf(xlow)
    xhigh = _to_mpf(xhigh)
    ylow = _to_mpf(ylow)
    yhigh = _to_mpf(yhigh)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    scaled_xticks, xticks = _get_based_ticks(xlow, xhigh, xbase, xtick)
    scaled_yticks, yticks = _get_based_ticks(ylow, yhigh, ybase, ytick)
    xticks_kwargs = {"rotation": 45, "horizontalalignment": 'right'} if rotate_xticklabels else {}
    axes.set_xticks(scaled_xticks, xticks, **xticks_kwargs)
    axes.set_yticks(scaled_yticks, yticks)
    for one_plot in plotting_data:
        assert len(one_plot) in [3, 4]
        if len(one_plot) == 3:
            xdata, ydata, color = one_plot
            kwargs = {}
        else:
            xdata, ydata, color, kwargs = one_plot
        clipped_x, clipped_y = general_lib.clip_pair_of_arrays(xdata, ydata, xlow, xhigh, ylow, yhigh, x_period)
        scaled_x = general_lib.rescale_array_to_float(clipped_x, xlow, xhigh)
        scaled_y = general_lib.rescale_array_to_float(clipped_y, ylow, yhigh)
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ""
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        if "markersize" not in kwargs:
            kwargs["markersize"] = 1
        axes.plot(scaled_x, scaled_y, color=color, **kwargs)


def get_parametric_curve_data_and_color(param_function, param_start: mp.mpf, param_end: mp.mpf, num_steps: int,
                                        color_name: str, convert_to_polar_coords: bool = False) -> tuple:
    """Returns (x_array, y_array, color, *args) to plot a parametric curve where args include additional dict of kwargs.

    :param param_function: a function that takes a parameter value, mpf, and returns a tuple (x, y)
        of mpf values of the coordinates.
    :param param_start: the starting value of the parameter.
    :param param_end: the end value of the parameter (always included).
    :param num_steps: the number of values from param_start to param_end including them.
    :param color_name: to be supplied to colors_lib.get_plotting_color(..).
    :param convert_to_polar_coords: if True, assume that param_function returns Cartesian coordinates and the plot
        is in the corresponding polar coordinates (angle, distance) displayed as Cartesian.
    """
    param_list = mp.linspace(param_start, param_end, num_steps)
    point_list = [param_function(param_value) for param_value in param_list]
    if convert_to_polar_coords:
        point_list = [(v, u) for x, y in point_list for u, v in [mp.polar(mp.mpc(x, y))]]
    x_array, y_array = tuple(zip(*point_list))
    x_array, y_array = np.array(x_array), np.array(y_array)
    # x_array does not have to be sorted, and if not, the connecting lines may mess up the plot.
    # sort x_array and reorder y_array accordingly:
    sorting_indices = x_array.argsort()
    x_array, y_array = x_array[sorting_indices], y_array[sorting_indices]
    return x_array, y_array, colors_lib.get_plotting_color(color_name), {"linestyle": "solid", "linewidth": 1}


def get_data_and_color(orbit_filename: str, color_name: str, *args) -> tuple:
    """Returns (x_array, y_array, color, *args) given the names of the orbit file and the color.

    args may include a dict of kwargs passed to axes.plot(..).
    """
    orbit = general_lib.unpickle(orbit_filename)
    return orbit[:, 1], orbit[:, 2], colors_lib.get_plotting_color(color_name), *args


def get_separatrix_data_and_color(orbit_filename: str, color_name: str, *args, front=None, back=None,
                                  flip_phi=None, period_phi=None) -> tuple:
    """Returns (x_array, y_array, color, *args) given the names of the separatrix orbit file and the color.

    args may include a dict of kwargs passed to axes.plot(..).
    :param front: either None or int; in the latter case keep only the front part of the separatrix
        (the one that approaches the hyperbolic point as built) and remove from the orbits all points with
        the point number (given by orbit[:,0,...]) < front.
    :param back: either None or int; in the latter case keep only the back part of the separatrix
        (removing the one that approaches the hyperbolic point as built); remove from the orbits all points with
        the point number (given by orbit[:,0,...]) > back.
    :param flip_phi: if not None, flip the orbit w.r. to the line phi = flip_phi.  If None, return orbit unchanged.
    :param period_phi: if not None, take the residue of all phi values w.r. to period_phi.
    (Assumes that the point numbers are synchronized in all orbits; if not, keeps max number of points per doubt.)
    TODO: remove practically duplicate points close to the hyperbolic.
    """
    orbit = general_lib.flip_phi_orbit(general_lib.remove_front_back_of_orbits(
        general_lib.unpickle(orbit_filename), front=front, back=back), flip_phi, period_phi)
    return orbit[:, 1, ...].ravel(), orbit[:, 2, ...].ravel(), colors_lib.get_plotting_color(color_name), *args
