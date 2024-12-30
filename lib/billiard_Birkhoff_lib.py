import inspect
import json
import os.path
import pathlib
import shutil

import mpmath as mp
import numpy as np
import pandas as pd
import general_lib


class ExtendedJSON_Encoder(json.JSONEncoder):
    """Allow for serialization of mpmath.mp, np arrays and pandas DataFrames, not necessarily invertible."""
    def default(self, obj):
        if isinstance(obj, mp.mpf):
            return repr(obj)  # Convert mpmath.mpf to its string representation
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to a list
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")  # Convert pandas DataFrame to a list of dictionaries
        return super().default(obj)


class BirkhoffBilliard_k_A(general_lib.AbstractBilliard):
    """This class represents a specific billiard boundary and the precision with which it is calculated.

    Since mpmath precision cannot yet be made part of a context object mpmath.mp
    per https://mpmath.org/doc/1.3.0/basics.html , one should use the instance with the appropriate precision value.
    This value is not validated since mpmath root solvers increase the working precision while searching for roots.
    The constructor sets the current mp.mp.dps to the specified precision_decimals.
    """
    def __init__(self, k: int, A_exact: str, precision_decimals: int, A_repr=None):
        self.precision_decimals = precision_decimals
        mp.mp.dps = precision_decimals
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        self.k_repr = str(k)
        self.k = mp.mpf(k)
        if A_repr is None:
            A_repr = A_exact
        if not (isinstance(A_exact, str) and isinstance(A_repr, str)):
            raise ValueError("A must be specified by a string")
        self.A_repr = A_repr
        self.A = mp.mpf(A_exact)
        # self.param_boundary = None  # will be defined later

    def set_working_precision(self) -> None:
        mp.mp.dps = self.precision_decimals

    def working_precision(self) -> int:
        return self.precision_decimals

    def h_d_and_deriv(self, theta):
        """Returns h, h', d, d'"""
        k_theta = self.k * theta
        A_cos_k_theta = self.A * mp.cos(k_theta)
        d = mp.acos(A_cos_k_theta) / self.k
        d_deriv = general_lib.deriv_acos(A_cos_k_theta) * self.A * (-mp.sin(k_theta))
        pi_k_minus_d = mp.pi / self.k - d
        h = mp.cos(pi_k_minus_d)
        h_deriv = mp.sin(pi_k_minus_d) * d_deriv
        return h, h_deriv, d, d_deriv

    def h_and_deriv(self, theta):
        """Returns h, h'"""
        h, h_deriv, _, _ = self.h_d_and_deriv(theta)
        return h,h_deriv

    def theoretical_invariant_curve(self, theta):
        """Returns phi and p for the theoretical invariant curve and the parameter value theta."""
        h, h_deriv, d, d_deriv = self.h_d_and_deriv(theta)
        phi = (theta + d)  % (2 * mp.pi)
        p = h * mp.cos(d) + h_deriv * mp.sin(d)
        return phi, p

    def filename(self, phi: mp.mpf, p: mp.mpf, n_decimals_in_name: int, num_iter: int) -> str:
        """The name of the file containing the orbit, without clipping info."""
        return (f"orbits/k{self.k_repr}A{self.A_repr}/orbit_k{self.k_repr}A{self.A_repr}"
                f"phi{mp.nstr(phi, n=n_decimals_in_name)}p{mp.nstr(p, n=n_decimals_in_name)}"
                f"prec{self.precision_decimals}n_iter{num_iter}.pkl")

    def dir_orbit_rectangle(self, phi, p, num_iter: int, rectangle, n_decimals_in_name_phi: int,
                            n_decimals_in_name_p=None) -> str:
        """The path to the directory containing the orbit with the clipping info, from the orbits root.

        :param phi: the initial value of phi; converted into mp.mpf under the current value of mp.dps.
        :param p: the initial value of p; converted into mp.mpf under the current value of mp.dps.
        :param num_iter: the number of iterations to make in the orbit.
        :param rectangle: the bounds of the rectangle for clipping, either None or [phi_low, phi_high, p_low, p_high].
            If None, no clipping.
        :param n_decimals_in_name_phi: The number of decimals of phi to use in the directory name.
        :param n_decimals_in_name_p: The number of decimals of p to use in the directory name;
            if None, use n_decimals_in_name_phi.
        :returns The path to the directory.
        """
        phi = mp.mpf(phi)
        p = mp.mpf(p)
        if n_decimals_in_name_p is None:
            n_decimals_in_name_p = n_decimals_in_name_phi
        rectangle_string = "unclipped" if rectangle is None else\
            f"phi{rectangle[0]}to{rectangle[1]}p{rectangle[2]}to{rectangle[3]}"
        return (f"orbits/k{self.k_repr}A{self.A_repr}/orbit_k{self.k_repr}A{self.A_repr}"
                f"phi{mp.nstr(phi, n=n_decimals_in_name_phi)}p{mp.nstr(p, n=n_decimals_in_name_p)}"
                f"prec{self.precision_decimals}n_iter{num_iter}/{rectangle_string}")

    def param_boundary(self):
         return general_lib.parametric_from_supporting(self.h_and_deriv)

    def compare_orbits(self, orbit1:np.array, orbit2:np.array):
        """Compare two orbits of identical shape, and find the max discrepancy.

        The precision used in the comparison, is the working precision of the instance.
        :param orbit1, orbit2: As produced by general_lib.calculate_orbit(..), an np.array with calculated_orbit[:, 0]
            being the point numbers, calculated_orbit[:, 1] the x coords, and calculated_orbit[:, 2] the y coords.
            Another possibility is orbit1, orbit2 being arrays of larger dimension, same shape, whose slices for given
            values of all dimensions >=2 are produced by general_lib.calculate_orbit(..) (the application being
            separatrices).
        :returns the max discrepancy.
        """
        # validate shapes and numbering (not complete):
        if (orbit1.shape != orbit2.shape) or (orbit1[0, 0, ...] != orbit2[0, 0, ...]).all() or (
                orbit1[-1, 0, ...] != orbit2[-1, 0, ...]).all():
            raise ValueError(f"Orbits that are compared, must be similar. Trying to compare orbits with the shapes\n"
                             f"{orbit1.shape, orbit2.shape} and numbering ranging between"
                             f" ({orbit1[0, 0], orbit1[-1, 0]}) and ({orbit2[0, 0], orbit2[-1, 0]})")
        self.set_working_precision()
        discrepancy_phi = general_lib.circled_discrepancy(orbit1[:, 1], orbit2[:, 1], 2 * mp.pi)
        discrepancy_p = np.abs(orbit1[:, 2] - orbit2[:, 2]).max()
        return max(discrepancy_phi, discrepancy_p)

    def set_orbit_around_point(self, orbit: np.array, point: np.array) -> np.array:
        """Change the given orbit into an equivalent one near point.


        :param orbit: as returned by calculate_orbit(.): np.array with rows (num_point,phi,p).
        :param point: np.array of the shape (2,), (phi, p).
        :returns an orbit equivalent to orbit with the values of phi centered near point.
        """
        orbit = orbit.copy()
        lower_bound_for_phi = point[0] - mp.pi
        orbit[:, 1] = (lower_bound_for_phi + (orbit[:, 1] - lower_bound_for_phi) % (2 * mp.pi))
        return orbit

    @staticmethod
    def build_orbit_by_blocks(k: int, A_exact: str,precision_main: int, precision_val: int,
                              initial_phi, initial_p, num_iter: int, block_size: int, rectangle,
                              n_decimals_in_name_phi: int, path_to_orbit_ambient: str, n_decimals_in_name_p=None) -> str:
        """Build and validate an orbit by blocks.

        :param k: Defines the boundary.
        :param A_exact: Defines the boundary, string.
        :param precision_main: The decimal precision to be used in calculating the orbit.
        :param precision_val: The decimal precision to be used in validating the orbit (normally > precision_main).
        :param initial_phi, initial_p: the initial values of phi and p, int or string.
        :param num_iter: the number of iterations.
        :param block_size: the size of one block for validation and clipping.
        :param rectangle: the bounds of the rectangle for clipping, either None or [phi_low, phi_high, p_low, p_high].
            If None, no clipping.
        :param n_decimals_in_name_phi: The number of decimals of phi to use in the directory name.
        :param path_to_orbit_ambient: path to the ambient directory that contains "orbits", the root of all orbit files.
        :param n_decimals_in_name_p: The number of decimals of p to use in the directory name;
            if None, use n_decimals_in_name_phi.
        :returns the path to the orbit.
        All working files are placed in the directory whose name is returned by dir_orbit_rectangle(..).
        If this directory already exists, attempt to continue the calculations broken in the middle.
        On successful completion, the directory contains orbit.pkl, misc_outputs.pkl and temporary files containing
        the last blocks at the main and the validation precision.
        """
        billiard_main = BirkhoffBilliard_k_A(k, A_exact, precision_main)
        billiard_val = BirkhoffBilliard_k_A(k, A_exact, precision_val)
        destination_dir = os.path.join(
            path_to_orbit_ambient,
            billiard_main.dir_orbit_rectangle(
                initial_phi, initial_p, num_iter, rectangle, n_decimals_in_name_phi, n_decimals_in_name_p))
        filepaths = {"clipped_orbit_previous": f"{destination_dir}/clipped_orbit_previous.pkl",
                     "clipped_orbit_current": f"{destination_dir}/orbit.pkl",
                     "main_block_previous": f"{destination_dir}/main_block_previous.pkl",
                     "validation_block_previous": f"{destination_dir}/validation_block_previous.pkl",
                     "main_block_current": f"{destination_dir}/main_block_current.pkl",
                     "validation_block_current": f"{destination_dir}/validation_block_current.pkl",
                     "misc_outputs": f"{destination_dir}/misc_outputs.pkl"}
        general_lib.create_validate_orbit_by_blocks(initial_phi, initial_p, num_iter, block_size, billiard_main,
                                                    billiard_val, rectangle, filepaths)
        return filepaths["clipped_orbit_current"]

    @staticmethod
    def extend_orbit_by_blocks(k: int, A_exact: str, precision_main: int, precision_val: int,
                               initial_phi, initial_p, num_iter: int, orig_num_iter: int, block_size: int, rectangle,
                               n_decimals_in_name_phi: int, path_to_orbit_ambient: str,
                               n_decimals_in_name_p=None) -> str:
        """Extend an existing orbit (that was created with keep_temp_files = True) and validate it by blocks.

        :param k: Defines the boundary.
        :param A_exact: Defines the boundary, string.
        :param precision_main: The decimal precision to be used in calculating the orbit.
        :param precision_val: The decimal precision to be used in validating the orbit (normally > precision_main).
        :param initial_phi, initial_p: the initial values of phi and p, int or string.
        :param num_iter: the number of iterations in the new orbit.
        :param orig_num_iter: the number of iterations in the original orbit (used to find the path to it).
        :param block_size: the size of one block for validation and clipping.
        :param rectangle: the bounds of the rectangle for clipping, either None or [phi_low, phi_high, p_low, p_high].
            If None, no clipping.
        :param n_decimals_in_name_phi: The number of decimals of phi to use in the directory name.
        :param path_to_orbit_ambient: path to the ambient directory that contains "orbits", the root of all orbit files.
        :param n_decimals_in_name_p: The number of decimals of p to use in the directory name;
            if None, use n_decimals_in_name_phi.
        :returns the path to the new orbit.
        All working files are placed in the directory whose name is returned by dir_orbit_rectangle(..).
        If this directory already exists, attempt to continue the calculations broken in the middle.
        If not, the directory of the specified existing orbit is copied into it; if the latter does not exist
        or is empty, raises an exception.
        On successful completion, the directory contains orbit.pkl, misc_outputs.pkl and temporary files containing
        the last blocks at the main and the validation precision.
        """
        billiard_main = BirkhoffBilliard_k_A(k, A_exact, precision_main)
        destination_dir = os.path.join(
            path_to_orbit_ambient,
            billiard_main.dir_orbit_rectangle(
                initial_phi, initial_p, num_iter, rectangle, n_decimals_in_name_phi, n_decimals_in_name_p))
        if not (os.path.isdir(destination_dir) and os.listdir(destination_dir)):  # extending has not yet started;
            # Start the extending:
            original_orbit_dir = os.path.join(
                path_to_orbit_ambient,
                billiard_main.dir_orbit_rectangle(
                    initial_phi, initial_p, orig_num_iter, rectangle, n_decimals_in_name_phi, n_decimals_in_name_p))
            if not (os.path.isdir(original_orbit_dir) and os.listdir(original_orbit_dir)):
                raise ValueError(
                    f"The directory of the original orbit does not exist or is empty:\n{original_orbit_dir}")
            shutil.copytree(original_orbit_dir, destination_dir, dirs_exist_ok=True)
        return BirkhoffBilliard_k_A.build_orbit_by_blocks(
            k, A_exact, precision_main, precision_val, initial_phi, initial_p, num_iter, block_size, rectangle,
            n_decimals_in_name_phi, path_to_orbit_ambient, n_decimals_in_name_p)

    separatrix_filename = "separatrix.pkl"
    separatrix_temp_filename = "temp_separatrix.pkl"
    separatrix_misc_outputs = "misc_outputs.pkl"
    separatrix_temp_misc_outputs = "temp_misc_outputs.pkl"

    def separatrix_dir_path(self, path_to_orbit_ambient: str, dir_name: str):
        return os.path.join(path_to_orbit_ambient, "orbits", f"k{self.k_repr}A{self.A_repr}", "separatrices", dir_name)

    @staticmethod
    def separatrix_by_blocks(
            k: int, A_exact: str, precision_main: int, precision_val: int, period: int, iteration_limit: int,
            hyperbolic_point, approach_separatrix_direction, other_separatrix_direction, precision,
            separating_coefficient, forward: bool, starting_point1, starting_point2, approaching_ratio: float,
            num_new_orbits: int, block_size: int, num_steps_back: int, path_to_orbit_ambient: str,
            dir_name: str, verbose: bool = False) -> str:
        """Build separatrix by blocks.

        :param k: Defines the boundary.
        :param A_exact: Defines the boundary, string.
        :param precision_main: The decimal precision to be used in calculating the orbit.
        :param precision_val: The decimal precision to be used in validating the orbit (normally > precision_main).
        :param period: the period of the hyperbolic point.
        :param iteration_limit: the limit on the number of iterations (of the billiard repeated period times, or
            period * iteration_limit billiard iterations) to make.
        :param hyperbolic_point: (phi, p) array-like of the hyperbolic point.
        :param approach_separatrix_direction: (phi, p) array-like of the eigenvector of the differential
            of the map that returns to the singular point; this is the eigenvector that indicates the direction from
            which we are approaching the singular point to find the separatrix.
        :param other_separatrix_direction: (phi, p) array-like of the eigenvector of the differential
            of the map that returns to the singular point; this is the other eigenvector.
        :param precision: An orbit that gets within this distance from singular_point, is considered to be separatrix.
            (As normally the product of the eigenvalues is 1, if the error in the direction of the separatrix is
            decreased N times after many iterations, the error in the other eigendirection - across the separatrix -
            is increased N times.  Thus, it is reasonable to start the calculations with ~ precision ** 2 to get to the
            desired precision of the singular point.)
        :param separating_coefficient: indicates how to split the angle between the eigenvectors to find the direction
            such that crossing it indicates missing the singular point.  The value 1 indicates splitting the angle
            in half, larger values indicate splitting closer to the approaching direction and smaller ones closer to
            other_separatrix_direction or its opposite.
        :param forward: if False, run the orbit backwards w.r. to the billiard.  (This is necessary if the eigenvalue of
            the differential in approach_separatrix_direction is > 1: then the forward direction of the billiard
            is away from the hyperbolic_point.)
        :param starting_point1: To find one orbit using orbit_hit_hyperbolic_point(.).
        :param starting_point2: To find one orbit using orbit_hit_hyperbolic_point(.).
        :param approaching_ratio: Each next orbit is built from starting_point1 and starting_point2 obtained
            from the previous values by multiplying their vectors from hyperbolic_point by approaching_ratio.
        :param num_new_orbits: the number of new orbits to create.
        :param block_size: The number of the new orbits to create in each block.
        :param num_steps_back: The number of steps away from the hyperbolic point to extend each of the new orbits.
            Do not extend if num_steps <= 0.
        :param path_to_orbit_ambient: path to the ambient directory that contains "orbits", the root of all orbit files.
        :param dir_name: the name of the directory (inside path_to_orbit_ambient) in which the separatrix, misc_outputs
            and the log files are placed.
        :param verbose: reports details in finding the starting point of each separatrix orbit;
            helpful if the eigenvalues are close to 1.
        :returns the path to the separatrix.
        :raises ValueError if the orbits that start from starting_point1 and starting_point2 at any step,
            pass at the same side of hyperbolic point.
        :raises RuntimeError if an orbit that starts at one of the binary search points
            (including starting_point1, starting_point2), passes behind the hyperbolic point, or
            if binary search get to the current mpmath precision limit without the result (this may happen either if
            the separatrix crosses the angle defined by separating_coefficient - in which case starting points need
            to be chosen closer to the hyperbolic point - or in the unlikely case that mp.mp.dps is insufficient).
        If the directory dir_name already exists, attempt to continue the calculations broken in the middle.
        On successful completion, the directory contains orbit.pkl, misc_outputs.pkl and
        <name_of_the_calling_script>.log; the latter contains the input data and the printout of misc_outputs.pkl.
        """
        param_dict = dict(inspect.currentframe().f_locals)
        billiard_main = BirkhoffBilliard_k_A(k, A_exact, precision_main)
        billiard_val = BirkhoffBilliard_k_A(k, A_exact, precision_val)
        destination_dir = billiard_main.separatrix_dir_path(path_to_orbit_ambient, dir_name)
        filepaths = {"separatrix": os.path.join(destination_dir, BirkhoffBilliard_k_A.separatrix_filename),
                     "temp_separatrix": os.path.join(destination_dir, BirkhoffBilliard_k_A.separatrix_temp_filename),
                     "misc_outputs": os.path.join(destination_dir, BirkhoffBilliard_k_A.separatrix_misc_outputs),
                     "temp_misc_outputs": os.path.join(
                         destination_dir, BirkhoffBilliard_k_A.separatrix_temp_misc_outputs)}
        max_discrepancy = general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            approaching_ratio, num_new_orbits, block_size, num_steps_back, filepaths, verbose)
        stack = inspect.stack()
        if len(stack) <= 1:
            raise RuntimeError("Unknown error: separatrix_by_blocks(.) has no previous stack frame.")
        calling_script_name = stack[1].filename
        data = {"params": param_dict, "max_discrepancy": max_discrepancy,
                "progress": general_lib.unpickle(filepaths["misc_outputs"])}
        with open(pathlib.Path(os.path.join(destination_dir, os.path.basename(calling_script_name))).with_suffix(
                ".log"), "w") as file:
            json.dump(data, file, cls=ExtendedJSON_Encoder, indent=4)
        return filepaths["separatrix"]

    @staticmethod
    def extend_validate_separatrix(k: int, A_exact: str, precision_main: int, precision_val: int, forward: bool,
                                   path_to_orbit_ambient: str, dir_name: str, num_steps: int) -> float:
        """Extend each of the orbits comprising the given separatrix, away from the hyperbolic point, and validate.

        Logs to <name_of_the_calling_script>.log .
        :param k: Defines the boundary.
        :param A_exact: Defines the boundary, string.
        :param precision_main: The decimal precision to be used in calculating the orbit.
        :param precision_val: The decimal precision to be used in validating the orbit (normally > precision_main).
        :param forward: if False, run the orbit backwards w.r. to the billiard.  (This is necessary if the eigenvalue of
            the differential in approach_separatrix_direction is > 1: then the forward direction of the billiard
            is away from the hyperbolic_point.)
        :param path_to_orbit_ambient: path to the ambient directory that contains "orbits", the root of all orbit files.
        :param dir_name: the name of the directory (inside path_to_orbit_ambient) in which the separatrix, misc_outputs
            and the log files are placed.
        :param num_steps: The number of steps away from the hyperbolic point to extend each of the orbits comprising
            separatrix_to_extend.  Do not extend if num_steps <= 0.
        :return: max_discrepancy between the calculation with billiard_main and billiard_val, float.
        """
        billiard_main = BirkhoffBilliard_k_A(k, A_exact, precision_main)
        billiard_val = BirkhoffBilliard_k_A(k, A_exact, precision_val)
        destination_dir = billiard_main.separatrix_dir_path(path_to_orbit_ambient, dir_name)
        separatrix_filepath = os.path.join(destination_dir, BirkhoffBilliard_k_A.separatrix_filename)
        stack = inspect.stack()
        if len(stack) <= 1:
            raise RuntimeError("Unknown error: extend_validate_separatrix(.) has no previous stack frame.")
        calling_script_name = stack[1].filename
        log_filepath = pathlib.Path(os.path.join(destination_dir, os.path.basename(calling_script_name))).with_suffix(
                ".log")
        return general_lib.extend_validate_separatrix_from_file(
            billiard_main, billiard_val, forward, separatrix_filepath, num_steps, log_filepath)
