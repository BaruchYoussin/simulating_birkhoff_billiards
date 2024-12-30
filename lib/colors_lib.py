import matplotlib.colors

_plotting_colors = ["b", "lime", "r", "c", "m", "k", "orange", "tab:brown"]
# _plotting_colors = ["b", "lime", "r", "xkcd:cyan", "xkcd:light magenta", "k", "orange", "tab:brown"]
_plotting_colormap = matplotlib.colors.ListedColormap(_plotting_colors)

_plotting_color_names = {"blue": 0, "green": 1, "red": 2, "cyan": 3, "magenta": 4, "black": 5, "orange": 6, "brown": 7}


def get_plotting_color(color_name: str) -> tuple:
    """Get a tuple of RGBA values for a listed name."""
    return _plotting_colormap(_plotting_color_names[color_name], bytes=False)
