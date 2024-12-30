import datetime
import filecmp
import gc
import itertools
import os.path
import pickle
import time

import mpmath as mp
import numpy as np
import pandas as pd


class AbstractBilliard:
    """This class represents an abstract billiard boundary.

    In case of high precision calculation, normally different instances should be created for different precisions.
    """
    def param_boundary(self):
        """Returns a function that takes the parameter value and returns the x, y coordinates of a plane point."""
        raise NotImplementedError("Not implemented.")

    def set_working_precision(self) -> None:
        raise NotImplementedError("Not implemented.")

    def working_precision(self) -> int:
        raise NotImplementedError("Not implemented.")

    def compare_orbits(self, orbit1: np.array, orbit2: np.array):
        """Compare two orbits of identical shape, and find the max discrepancy.

        The precision used in the comparison, is the working precision of the instance.
        """
        raise NotImplementedError("Not implemented.")

    def set_orbit_around_point(self, orbit: np.array, point: np.array) -> np.array:
        """In case there is periodicity in the phase space, change the given orbit into an equivalent one near point.

        :param orbit: as returned by calculate_orbit(.): np.array with rows (num_point,phi,p).
        :param point: np.array of the shape (2,), (phi, p).
        :returns an orbit equivalent to orbit with the values of phi, p centered near point.
        """
        raise NotImplementedError("Not implemented.")


def parametric_from_supporting(supporting_function_with_derivative):
    """Construct parametric representation of a curve given its supporting function.

    :param supporting_function_with_derivative: a functin that takes mpmath.mpf and returns
        (supporting_function, its_derivative), (mpmath.mpf, mpmath.mpf).
    :returns a function that takes the parameter value and returns the x, y coordinates of a plane point.
    """
    def parametric(parameter_value):
        value, derivative = supporting_function_with_derivative(parameter_value)
        complex_result = value + derivative * mp.j
        complex_result *= mp.exp(mp.j * parameter_value)
        return complex_result.real, complex_result.imag

    return parametric


def deriv_acos(x):
    return mp.mpf(-1) / mp.sqrt(mp.mpf(1) - x * x)


def projection_of_point_onto_direction(point: tuple, direction: mp.mpf) -> mp.mpf:
    """Returns the projection of a given point onto a given direction."""
    exp = mp.exp(mp.j * direction)
    return (mp.matrix(point).T * mp.matrix([exp.real, exp.imag]))[0,0]


def intersect_line_with_parametric_billiard_boundary(parametric_boundary, phi:mp.mpf, p:mp.mpf,
                                                     tolerance=None, forward: bool = True) -> mp.mpf:
    """Finds the left intersection point of a directed straight line with a parametric billiard boundary.

    :param parametric_boundary: a function mp.mpf -> (mp.mpf, mp.mpf) parametrizing the boundary as a function
        of the direction of its tangent given as the signed angle to the Ox axis.
    :param phi: The signed angle between the Ox axis and the normal to the line which is counterclockwise
        from the line direction.
    :param p: The signed distance from the origin to the line in the direction of the above normal.
    :param tolerance: The tolerance for the squared error to be used in the numeric root finding.
        Default: 10 ** (-2 * mp.mp.dps).
        (Since findroot temporarily increases dps by 6, this assumes that the derivative of the error function
            is < 10 ** 6.)
    :param forward: if False, go backward along the direction of the given line.
    :returns The value of the parameter corresponding to the second (if forward = True) or first (if forward = False)
        intersection point of the line with the boundary.
    """
    if tolerance is None:
        tolerance = mp.mpf(10) ** (-2 * mp.mp.dps)
    return mp.findroot(
        lambda parameter_value: projection_of_point_onto_direction(parametric_boundary(parameter_value), phi) - p,
        (phi, phi + mp.pi) if forward else (phi - mp.pi, phi), solver="pegasus", tol=tolerance)


def reflect_line_at_point(parameter_value_of_point: mp.mpf, direction_of_line: mp.mpf) -> mp.mpf:
    """Reflect a directed straight line w.r. to a billiard boundary parametrized by its tangent direction.

    :param parameter_value_of_point: The parameter value of the reflection point on the boundary (which should lie
        on the reflected line).
    :param direction_of_line: The signed angle from the Ox axis to the line direction.
    :returns The direction of the reflected line = (2 * parameter_value_of_point - direction_of_line) modulo 2 pi.
    """
    return (2 * parameter_value_of_point - direction_of_line) % (2 * mp.pi)


def reflect_line_at_boundary(phi: mp.mpf, p:mp.mpf, parametric_boundary, tolerance=None, forward: bool = True) -> tuple:
    """Reflect a straight line w.r. to the billiard boundary.

    :param phi: The signed angle between the Ox axis and the normal to the line which is counterclockwise
        from the line direction.
    :param p: The signed distance from the origin to the line in the direction of the above normal.
    :param parametric_boundary: a function mp.mpf -> (mp.mpf, mp.mpf) parametrizing the boundary as a function
        of the direction of its tangent given as the signed angle to the Ox axis.
    :param tolerance: The tolerance for the squared error to be used in the numeric root finding.
        Default: 10 ** (-2 * mp.mp.dps).
        (Since findroot temporarily increases dps by 6, this assumes that the derivative of the error function
            is < 10 ** 6.)
    :param forward: if False, go backward along the direction of the line.
    :returns (phi, p) of the reflected line.
    """
    parameter_value_at_intersection = intersect_line_with_parametric_billiard_boundary(
        parametric_boundary, phi=phi, p=p, tolerance=tolerance, forward=forward)
    reflected_phi = reflect_line_at_point(parameter_value_at_intersection, phi)
    reflected_p = projection_of_point_onto_direction(parametric_boundary(parameter_value_at_intersection),
                                                     reflected_phi)
    return reflected_phi, reflected_p


def iterate_function(fn, init_value, num_iter: int, *args, **kwargs) -> list:
    """Apply a function iteratively and return a list of all the results.

    :param fn: The function to apply; the initial and repeated values are substituted in its first parameter, unless
        init_value is a tuple, in which case it is unpacked into the first parameters.
    :param init_value: The initial value, possibly a tuple.
    :param num_iter: The number of iterations to apply.
    :param args: Additional positional arguments to supply to fn after the first parameter.
    :param kwargs: Additional keyword arguments to supply to fn.
    :returns The list [init_value, first result, ...], the total of num_iter + 1 elements.
    """
    def iterator_fn():
        nonlocal init_value
        for n in range(num_iter):
            yield init_value
            init_value = fn(*(init_value if isinstance(init_value, tuple) else (init_value,)), *args, **kwargs)
        yield init_value

    return list(iterator_fn())


def iterate_function_until_condition(fn, init_value, condition_fn, num_iter, *args, **kwargs) -> tuple:
    """Apply a function iteratively until the condition function returns non-None.

    :param fn: The function to apply, takes an object similar to init_value and returns a similar object.
    :param init_value: The initial value, an array if contains more than one value.
    :param condition_fn: a function on objects similar to init_value; fn is applied iteratively until
        condition_fn returns non-None.
    :param num_iter: The limit on the number of iterations to apply.
    :param args: Additional positional arguments to supply to fn after the first parameter.
    :param kwargs: Additional keyword arguments to supply to fn.
    :returns returned_from_condition_function, [init_value, first result, ...] where returned_from_condition_function
        is the first non-None returned by condition_fn.
        In case num_iter has been reached and condition_fn returned only None, returns returned_from_condition_function = None.
    """
    def iterator_fn():
        nonlocal init_value
        for n in range(num_iter):
            yield init_value
            init_value = fn(init_value, *args, **kwargs)
            ret = condition_fn(init_value)
            if ret is not None:
                yield init_value
                return ret
        yield init_value

    gen = iterator_fn()
    results = []
    returned_from_condition_function = None  # this is already assumed by the language and is not necessary;
        # keep for readability.
    while True:
        try:
            results.append(next(gen))
        except StopIteration as e:
            returned_from_condition_function = e.value
            break
    return returned_from_condition_function, results


def calculate_orbit(billiard: AbstractBilliard, initial_phi: mp.mpf, initial_p:mp.mpf,
                    num_iterations: int, start_numbering: int = 0, forward: bool = True) -> np.array:
    """Create orbit as np.array with rows (num_point,phi,p) starting with (start_numbering, initial_phi, initial_p).

    The orbit is returned as a np.array of type Object.
    The working precision is set by the billiard argument and is not restored.
    if forward = False, go backward along the billiard.

    """
    billiard.set_working_precision()
    parametric_boundary = billiard.param_boundary()
    initial_phi = mp.mpf(initial_phi)
    initial_p = mp.mpf(initial_p)
    result_list = iterate_function(
        lambda n, phi, p, forward: (n + 1, *reflect_line_at_boundary(phi, p, parametric_boundary, forward=forward)),
        init_value=(start_numbering, initial_phi, initial_p), num_iter=num_iterations, forward=forward)
    return np.array(result_list)


def unpickle(filename):
    """Unpickle a pkl file"""
    with open(filename, "rb") as file:
        return pickle.load(file)


def pickle_one_object(object, filename):
    dirname = os.path.dirname(filename)
    if dirname is not None and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(object, file)


def linspace(a: mp.mpf, b: mp.mpf, n: int) -> list:
    """An improved version of mpmath.linspace with 3 args: replaces numeric zero by actual zero."""
    a, b = mp.mpf(a), mp.mpf(b)
    numeric_bound = max([mp.fabs(a), mp.fabs(b)]) * (10 ** (-mp.mp.dps))
    return [0 if mp.fabs(x) < numeric_bound else x for x in mp.linspace(a, b, n)]


def two_sided_range(a: mp.mpf, b: mp.mpf, step: mp.mpf) -> list:
    """The list of all numbers divisible by step between a and b, both including up to numerical zero."""
    a, b, step = (mp.mpf(x) for x in [a, b, step])
    numeric_bound = max([mp.fabs(a), mp.fabs(b)]) * (10 ** (-mp.mp.dps + 0.5))
    return [n * step for n in
            mp.arange(mp.ceil((a - numeric_bound)/step), mp.floor((b + numeric_bound)/step) + 1)]


def get_residues_to_left_bound(x: np.ndarray, left_bound: mp.mpf, period: mp.mpf) -> np.ndarray:
    """Change all values in x by multiples of period in such way that each lies in [left_bound, left_bound + period).
    """
    return ((x - left_bound) % period) + left_bound


def clip_pair_of_arrays(x: np.array, y: np.array, xlow, xhigh, ylow, yhigh, x_period=None) -> tuple:
    """Clip a pair of numeric arrays (considered as a list of points in the plane) by a rectangle.

    :param x: the array of the x coordinates, 1-dim.
    :param y: the array of the y coordinates, 1-dim of the same length as x.
    :param xlow: the lower limit to clip on the x axis
    :param xhigh: the upper limit to clip on the x axis
    :param ylow: the lower limit to clip on the y axis
    :param yhigh: the upper limit to clip on the y axis
    :param x_period: if not None, understood as the period in the x axis.
        All x values are changed by the multiples of x_period so that they belong in the interval [xlow, xlow + period).
        (In this way 0 may be between xlow and xhigh and all x values will be clipped into this interval correctly
        by subtracting x_period from them if necessary.)
    :returns x_coords, y_coords of the points that lie in the rectangle (possibly as views on the original x, y).

    """
    if x_period is not None:
        x = get_residues_to_left_bound(x, xlow, x_period)
    condition_array = (x >= xlow) & (x <= xhigh) & (y >= ylow) & (y <= yhigh)
    return x.compress(condition_array, axis=0), y.compress(condition_array, axis=0)


def clip_orbit(calculated_orbit, xlow, xhigh, ylow, yhigh) -> np.array:
    """Clip orbit by a rectangle.

    :param calculated_orbit: As produced by calculate_orbit(..), an np.array with calculated_orbit[:, 0] being
        the point numbers, calculated_orbit[:, 1] the x coords, and calculated_orbit[:, 2] the y coords.
    :param xlow: the lower limit to clip on the x axis
    :param xhigh: the upper limit to clip on the x axis
    :param ylow: the lower limit to clip on the y axis
    :param yhigh: the upper limit to clip on the y axis
    :returns the clipped orbit, in the same format.
    """
    x = calculated_orbit[:, 1]
    y = calculated_orbit[:, 2]
    condition_array = (x >= xlow) & (x <= xhigh) & (y >= ylow) & (y <= yhigh)
    return calculated_orbit.compress(condition_array, axis=0)


def extend_orbit(calculated_orbit: np.array, extension_info) -> np.array:
    """Append a calculated orbit to previous data.

    No check is made that the precision of the appended data corresponds to the precision of the previous data
    as calculated_orbit and the returned array are of type Object which can mix different types in different cells,
    and in particular, different precisions.
    :param calculated_orbit: np.array produced by calculate_orbit; starts with
        [initial_num_point, initial_phi, initial_p].
    :param extension_info: None or dict with two entries:
        "previous_points": the array of previous points; if None, then this array is taken to have one element,
            [initial_num_point, initial_phi, initial_p].
        "rectangle": None or [xlow, xhigh, ylow, yhigh] to clip the calculated orbit (after the initial point)
            before appending it to the array of previous points. If None, no clipping.
            xlow, xhigh, ylow, yhigh are converted to mp.mpf under the current value of mp.mp.dps;
            thus, they can and should be passed as strings.
        If None is supplied instead of a dict, it is understood as {"previous_points": None, "rectangle": None}.
    :returns np.array containing the previous points (not clipped) and the new points, clipped as specified.
    """
    if extension_info is None:
        return calculated_orbit
    if extension_info["previous_points"] is None:
        extension_info["previous_points"] = calculated_orbit[:1, :]
    if extension_info["rectangle"] is None:
        new_points = calculated_orbit[1:, :]
    else:
        rectangle = [mp.mpf(bound) for bound in extension_info["rectangle"]]
        new_points = clip_orbit(calculated_orbit[1:, :], *rectangle)
    return np.concatenate((extension_info["previous_points"], new_points), axis=0)


def rescale_array(array: np.ndarray, low, high) -> np.ndarray:
    """Rescale an array making the low value equal 0 and the high value equal 1."""
    return (array - low) / (high - low)


def rescale_array_to_float(array: np.ndarray, low, high) -> np.ndarray:
    """Rescale an array making the low value equal 0 and the high value equal 1, and converting all values to float."""
    return rescale_array(array, low, high).astype(float)


def circled_discrepancy(array_1, array_2, modulus):
    """Returns the max difference between array_1 and array_2 (must be of the same shape) modulo modulus."""
    diff = np.abs((array_2 % modulus)  - (array_1 % modulus))
    return np.minimum(diff, modulus - diff).max()


def circled_signed_discrepancy(first, second, modulus):
    """Signed distance from first to second where both understood modulo modulus; modulus=None is understood as inf."""
    if modulus is None:
        return second - first
    distance = (second - first) % modulus
    if distance > modulus / 2:
        return distance - modulus
    if distance < - modulus / 2:
        return distance + modulus
    return distance


def create_validate_orbit_by_blocks(initial_phi, initial_p, num_iter: int, block_size: int,
                                    billiard_main: AbstractBilliard, billiard_val: AbstractBilliard,
                                    rectangle, filepaths: dict, keep_temp_files: bool = True) -> None:
    """Create, validate, clip and write to disk an orbit by blocks of specified size.
    
    :param initial_phi, initial_p: the initial values of phi and p, int or string.
    :param num_iter: the number of iterations.
    :param block_size: the size of one block for validation and clipping.
    :param billiard_main: an instance of AbstractBilliard representing the boundary with the precision
        to be used to calculate the output orbit.
    :param billiard_val: an instance of AbstractBilliard representing the boundary with the validation precision
        (which is assumed to be greater than the main working precision).
    :param rectangle: the bounds of the rectangle for clipping, either None or [phi_low, phi_high, p_low, p_high].
        If None, no clipping.
        phi_low, phi_high, p_low, p_high are converted to mp.mpf under the current value of mp.mp.dps;
        thus, they can and should be passed as strings.
    :param filepaths: a dict of filepaths with the following keys:
        "clipped_orbit_previous": where the previous version of the final orbit is written, whether clipped or not
            (a temporary file).
        "clipped_orbit_current": where the current version of the final orbit is written, whether clipped or not.
            The final version is written in this file on exit.
        "main_block_previous": a temporary file containing the previous block calculated with the precision 
            main_decimals.
        "validation_block_previous": a temporary file containing the previous block calculated with the precision 
            validation_decimals.
        "main_block_current": a temporary file containing the current block calculated with the precision 
            main_decimals.
        "validation_block_current": a temporary file containing the current block calculated with the precision 
            validation_decimals.
        "misc_outputs": a file containing the table of various outputs including validation discrepancies.
            The table has one row for each block and the following columns:
                block_start, block_end: the point numbers in the blocks, both including
                    (they overlap in adjacent blocks: block_end of the previous block is block_start of the next one).
                perf_time_main, process_time_main: the times spend in creating the block at the main precision, hours.
                perf_time_val, process_time_val:  the times spend in creating the block at the main precision, hours.
                discrepancy: between the main and the val orbit, as returned by compare_orbits(.).
    :param keep_temp_files: if True, do not remove the temporary files (main_block_current, validation_block_current).
    On exit, the working precision is at the validation level (as set in billiard_val).
    """
    current_files = [filepaths[name] for name in
                     ["clipped_orbit_current", "main_block_current", "validation_block_current"]]
    previous_files = [filepaths[name] for name in
                      ["clipped_orbit_previous", "main_block_previous", "validation_block_previous"]]
    # Check which files are present, and where to continue
    billiard_val.set_working_precision()  # set working precision to the validation level, in case loading previous
    # files and extracting from them the initial values for the calculations to be made, are affected
    # by the working precision.
    if os.path.exists(filepaths["misc_outputs"]):
        misc_outputs_table = unpickle(filepaths["misc_outputs"])
        if len(misc_outputs_table) == 0:
            raise RuntimeError(f"Invalid initial table:\n{misc_outputs_table}")
        last_row = misc_outputs_table.iloc[-1]
        block_start = last_row.name
        if last_row.isna().loc["block_end"]:
            raise RuntimeError(f"Invalid last row in the table:\n{last_row}")
        block_end = last_row.loc["block_end"]
        if last_row.isna()[["perf_time_main", "process_time_main"]].any():
            raise RuntimeError(f"Invalid last row in the table:\n{last_row}")
        if last_row.isna().any():
            if not os.path.exists(filepaths["main_block_current"]):
                raise RuntimeError(f"The main_block_current file missing.")
            clipped_orbit_done = os.path.exists(filepaths["clipped_orbit_current"])
            val_block_done = not last_row.isna()[["perf_time_val", "process_time_val"]].any()
            if val_block_done and not os.path.exists(filepaths["validation_block_current"]):
                raise RuntimeError(f"The validation_block_current file missing.")
            if clipped_orbit_done:
                if val_block_done:
                    starting_step = 4
                else:
                    starting_step = 3
            else:
                starting_step = 2
        else:  # the last row is complete.
            # We are either done, or in the end of the loop, with all current files present,
            # and possibly with previous if not yet deleted:
            if block_end >= num_iter:
                raise RuntimeError(f"The table is complete, we are apparently done.")
            current_files_exist = [os.path.exists(filename) for filename in current_files]
            # If all of them exist, starting_step = 5, if none, starting_step = 1 in the next pass of the loop,
            # otherwise some of them have been renamed into previous and others not, or some other error:
            if all(current_files_exist):
                starting_step = 5
            elif not any(current_files_exist):
                starting_step = 1
            else:
                raise RuntimeError("Cannot resolve current/previous files at the end of the loop pass:"
                                   " some of the current files exist and some not.")
        # find the starting values of phi and p:
        if starting_step <= 1:  # the first row of the table is complete.  Advance to the next one:
            block_start = block_end
            block_end += block_size
            misc_outputs_table.at[block_start, "block_end"] = block_end
            assert os.path.exists(
                filepaths["main_block_previous"]), f"main_block_previous file expected but not found"
            main_block_previous = unpickle(filepaths["main_block_previous"])
            last_point = main_block_previous[-1, :]
            assert last_point[0] == block_start
            block_start_phi_main = last_point[1]
            block_start_p_main = last_point[2]
            del last_point, main_block_previous
            gc.collect()
        else:
            assert os.path.exists(
                filepaths["main_block_current"]), f"main_block_current file expected but not found"
            main_orbit = unpickle(filepaths["main_block_current"])
            last_point = main_orbit[-1, :]
            assert last_point[0] == block_end
            block_start_phi_main = last_point[1]
            block_start_p_main = last_point[2]
            del last_point
            if starting_step > 2:
                del main_orbit
                gc.collect()
        if starting_step <= 3:
            if block_start == 0:
                block_start_phi_val = initial_phi
                block_start_p_val = initial_p
            else:
                assert os.path.exists(
                    filepaths["validation_block_previous"]), f"validation_block_previous file expected but not found"
                val_block_previous = unpickle(filepaths["validation_block_previous"])
                last_point = val_block_previous[-1, :]
                assert last_point[0] == block_start
                block_start_phi_val = last_point[1]
                block_start_p_val = last_point[2]
                del last_point, val_block_previous
                gc.collect()
        else:
            val_orbit = unpickle(filepaths["validation_block_current"])
            last_point = val_orbit[-1, :]
            assert last_point[0] == block_end
            block_start_phi_val = last_point[1]
            block_start_p_val = last_point[2]
            del last_point
            if starting_step > 4:
                del val_orbit
                gc.collect()
    else:  # initialization if no previous orbits
        block_start = 0
        block_end = block_size
        misc_outputs_table = pd.DataFrame.from_dict(
            {"block_start": [block_start], "block_end": [block_end], "perf_time_main": [None],
             "process_time_main": [None], "perf_time_val": [None], "process_time_val": [None],
             "discrepancy": [None]}).set_index("block_start")
        starting_step = 0
        block_start_phi_main = block_start_phi_val = initial_phi
        block_start_p_main = block_start_p_val = initial_p
    # else if there are previous orbits, load them and get the initial values from them
    # and load misc_outputs_table
    while True:  # in the beginning of the cycle, the current files are absent
        if starting_step <= 1:  # calculate orbit with main precision:
            perf_counter_start = time.perf_counter_ns()
            process_counter_start = time.process_time_ns()
            main_orbit = calculate_orbit(billiard_main, block_start_phi_main, block_start_p_main, block_size,
                                         block_start)
            # The current working precision has been set to that of billiard_main
            perf_counter_end = time.perf_counter_ns()
            process_counter_end = time.process_time_ns()
            perf_time = (perf_counter_end - perf_counter_start) / 3.6e12  # hours
            process_time = (process_counter_end - process_counter_start) / 3.6e12
            print(f"Calculated the orbit, iterations {block_start}-{block_end},"
                  f" {billiard_main.working_precision()} decimals,"
                  f" perf_time: {perf_time:.3f}, process_time: {process_time:.3f} hours", flush=True)
            pickle_one_object(main_orbit, filepaths["main_block_current"])
            print(f"Orbit pickled successfully at {filepaths['main_block_current']}", flush=True)
            misc_outputs_table.at[block_start, "perf_time_main"] = perf_time
            misc_outputs_table.at[block_start, "process_time_main"] = process_time
            pickle_one_object(misc_outputs_table, filepaths["misc_outputs"])
            last_point = main_orbit[-1, :]
            assert last_point[0] == block_end
            block_start_phi_main = last_point[1]
            block_start_p_main = last_point[2]
        if starting_step <= 2:  # extend and clip:
            if block_start == 0:
                previous_points = None
            else:
                assert os.path.exists(
                    filepaths["clipped_orbit_previous"]), f"clipped_orbit_previous file expected but not found"
                previous_points = unpickle(filepaths["clipped_orbit_previous"])
            extension_info = {"previous_points": previous_points, "rectangle": rectangle}
            extended_clipped_orbit = extend_orbit(main_orbit, extension_info)
            pickle_one_object(extended_clipped_orbit, filepaths["clipped_orbit_current"])
            del main_orbit, previous_points, extended_clipped_orbit, extension_info
            gc.collect()
        if starting_step <= 3:  # calculate orbit with validation precision:
            perf_counter_start = time.perf_counter_ns()
            process_counter_start = time.process_time_ns()
            val_orbit = calculate_orbit(billiard_val, block_start_phi_val, block_start_p_val,
                                               block_size, block_start)
            perf_counter_end = time.perf_counter_ns()
            process_counter_end = time.process_time_ns()
            perf_time = (perf_counter_end - perf_counter_start) / 3.6e12  # hours
            process_time = (process_counter_end - process_counter_start) / 3.6e12
            print(f"Calculated the orbit, iterations {block_start}-{block_end},"
                  f" {billiard_val.working_precision()} decimals,"
                  f" perf_time: {perf_time:.3f}, process_time: {process_time:.3f} hours", flush=True)
            pickle_one_object(val_orbit, filepaths["validation_block_current"])
            print(f"Orbit pickled successfully at {filepaths['validation_block_current']}", flush=True)
            misc_outputs_table.at[block_start, "perf_time_val"] = perf_time
            misc_outputs_table.at[block_start, "process_time_val"] = process_time
            pickle_one_object(misc_outputs_table, filepaths["misc_outputs"])
            last_point = val_orbit[-1, :]
            assert last_point[0] == block_end
            block_start_phi_val = last_point[1]
            block_start_p_val = last_point[2]
            del last_point
        if starting_step <= 4:  # Compare the main and the validation orbits
            # The working precision is kept at the validation level since the last call to calculate_orbit(..).
            main_orbit = unpickle(filepaths["main_block_current"])
            discrepancy = billiard_main.compare_orbits(main_orbit, val_orbit)
            del main_orbit, val_orbit
            gc.collect()
            misc_outputs_table.at[block_start, "discrepancy"] = discrepancy
            pickle_one_object(misc_outputs_table, filepaths["misc_outputs"])
            print(f"Discrepancy over iterations {block_start}-{block_end}: {mp.nstr(discrepancy, 3)}", flush=True)
        if starting_step <= 5:  # delete the previous files and rename current into previous:
            for filepath in [filepaths[name] for name in [
                "clipped_orbit_previous", "main_block_previous", "validation_block_previous"]]:
                if os.path.exists(filepath):
                    os.remove(filepath)
            print("Previous files deleted", flush=True)
            if block_end >= num_iter:  # done
                break  # Recall that the working precision is kept at the validation level, see above.
            current_previous_pairs = [(filepaths[name1], filepaths[name2]) for (name1, name2) in [
                ("clipped_orbit_current", "clipped_orbit_previous"),
                ("main_block_current", "main_block_previous"),
                ("validation_block_current", "validation_block_previous")]]
            for filepath_current, filepath_previous in current_previous_pairs:
                if os.path.exists(filepath_current):
                    os.rename(filepath_current, filepath_previous)
            print("Current files renamed into the previous ones")
            # prepare for the next loop
            block_start = block_end
            block_end += block_size
            misc_outputs_table.at[block_start, "block_end"] = block_end
            print(f"Starting the block {block_start}-{block_end}", flush=True)
        starting_step = 0
        # end of the loop
    # Recall that the working precision is kept at the validation level on the exit of the loop, see above.
    # remove temporary files (only current as previous have already been removed before the break):
    if not keep_temp_files:
        for filepath in [filepaths[name] for name in ["main_block_current", "validation_block_current"]]:
            if os.path.exists(filepath):
                os.remove(filepath)
    print(f"Clipped orbit complete: {filepaths['clipped_orbit_current']}")
    print(f"Max discrepancy: {mp.nstr(misc_outputs_table.discrepancy.max(), 3)}", flush=True)


def count_num_of_full_circles(points: np.ndarray) -> int:
    """Count the number of full circles that the given points take, assuming they increase, going over a circle.

    Assume that the given numbers specify points on a circle (as residues w.r. to the circle length) and they move
    in the increasing direction; any case of decrease is understood as passing through the right boundary and
    coming back from the left one.
    """
    points = np.array(points)
    if len(points.shape) != 1 or points.shape[0] == 0:
        raise ValueError(f"The point set for counting the number of full circles is of incorrect shape: {points}")
    if points.shape[0] == 1:
        return 0
    num_passes_beyond_the_right_boundary = (np.diff(points) < 0).sum()
    # each pass is a full circle except possibly the last one which is a full circle only if the endpoint is
    # to the right of the starting point; if num_passes_beyond_the_right_boundary == 0 then
    # the endpoint must be to the right of the starting point:
    return int(num_passes_beyond_the_right_boundary - 1 + ((points[-1] - points[0]) >= 0))


def circular_length(points: np.ndarray, period):
    """The distance from the 0th to the last points of an array, assuming its elements increase, going over a circle.

    Assume that the given numbers specify points on a circle (as residues w.r. to the circle length) and they move
    in the increasing direction; any case of decrease is understood as passing through the right boundary and
    coming back from the left one.
    The distance is measured in the positive direction, passing through all the points;
    if this involves going over the circle, this adds period = circle_length each time.
    In case the given numbers are outside [0, period), their remainder w.r. to period is taken
    (period is assumed to be positive).
    """
    points = np.array(points) % period
    num_circles = count_num_of_full_circles(points)
    vector_length = points[-1] - points[0]
    if vector_length < 0:
        num_circles += 1
    return num_circles * period + vector_length


def periodic_orbit_error(billiard: AbstractBilliard, num_iter: int, phi: mp.mpf, p: mp.mpf,
                         period_in_phi=None) -> tuple:
    """The error function for finding a periodic orbit using a nearby starting point.

    :param billiard: the billiard for which we solve for the periodic orbit.
    :param num_iter: the length of the periodic orbit we are looking for.
    :param phi: an approximation for the phi of a point on a periodic orbit.
    :param p: an approximation for the p of a point on a periodic orbit.
    :param period_in_phi: If not None, then the values of phi that differ in period_in_phi, are considered equal.
    :returns The difference (phi, p) between the original point and num_iter iterations of it.  The difference is
        calculated without taking into account the number of circles the orbit has made, and for this reason this
        function is appropriate only when (phi, p) is close to the actual periodic orbit.
    """
    orbit = calculate_orbit(billiard, phi, p, num_iter)
    last_point = orbit[num_iter, :]
    return circled_signed_discrepancy(phi, last_point[1], period_in_phi), last_point[2] - p


def periodic_orbit_error_circular(
        billiard: AbstractBilliard, num_iter: int, phi: mp.mpf, p: mp.mpf, period_in_phi,
        target_num_circles_phi: int) -> tuple:
    """The error function for finding a periodic orbit even for far away starting points.

    :param billiard: the billiard for which we solve for the periodic orbit.
    :param num_iter: the length of the periodic orbit we are looking for.
    :param phi: an approximation for the phi of a point on a periodic orbit.
    :param p: an approximation for the p of a point on a periodic orbit.
    :param period_in_phi: The values of phi that differ in period_in_phi, are considered equal;
        this is the period that is used for circular length.
    :param target_num_circles_phi: the number of the circles in the periodic orbit to be found
        (e.g., 1 if the rotation number is 7, and 3 if the rotation number is 7 1/3).
    :returns The difference (phi, p) between the original point and num_iter iterations of it,
        where the difference in the phi coordinate is circular length.
    """
    orbit = calculate_orbit(billiard, phi, p, num_iter)
    last_point = orbit[num_iter, :]
    phi_points = orbit[:, 1]
    return circular_length(phi_points, period_in_phi) - target_num_circles_phi * period_in_phi, last_point[2] - p


def periodic_in_phi_only_orbit(billiard: AbstractBilliard, num_iter: int, initial_phi, p_lower, p_upper,
                               period_in_phi, target_num_circles_phi: int, solver="pegasus", max_steps=300,
                               add_decimals_in_tolerance: int = -1) -> tuple:
    """Find a value of p for which the orbit that starts from (initial_phi, p) returns to initial_phi after num_iter.

    :param billiard: the instance of AbstractBilliard that defines the billiard map.
    :param num_iter: the number of iterations in the period.
    :param initial_phi: the initial value of phi to start with, mp.mpf or convertible to it.
    :param p_lower: the lower bound for the p to be found.
    :param p_upper: the upper bound for the p to be found.
    :param period_in_phi: The values of phi that differ in period_in_phi, are considered equal.
    :param target_num_circles_phi: the number of the circles in the periodic orbit to be found
        (e.g., 1 if the rotation number is 7, and 3 if the rotation number is 7 1/3).
    :param solver: to use in finding p.
    :param max_steps: the maximal number of steps in finding the p as a root of a nonlinear equation.
    :param add_decimals_in_tolerance: the number of decimals to be added to the minimal value in tolerance,
        see the code.
    :returns p, increase_in_p: both as mpmath.mpf. increase_in_p is the difference in p after applying num_iter.
    """

    def gap_in_phi(p):
        return periodic_orbit_error_circular(billiard, num_iter, initial_phi, p, period_in_phi,
                                             target_num_circles_phi)[0]

    p = mp.findroot(gap_in_phi, (p_lower, p_upper), solver=solver,
                    # tolerance cannot be 10**(-2*precision_decimals) as in
                    # intersect_line_with_parametric_billiard_boundary for the following reason:
                    # tol for all illinois algorithms is checked twice, first in the approximation loop which ends
                    # when the check is
                    # abs(the function value) < tol, and second, after the loop ends
                    # (whether the above condition is satisfied or max_iter is achieved) when the check is
                    # (the function value) ** 2 < tol.
                    # In case of intersect_line_with_parametric_billiard_boundary the first test succeeds after
                    # a small number of iterations as the function value == 0.
                    # In this case it does not work out, and loop always runs to max_iter.
                    tol=mp.mpf(10) ** (-billiard.working_precision() + add_decimals_in_tolerance), maxsteps=max_steps)
    increase_in_p = periodic_orbit_error_circular(billiard, num_iter, initial_phi, p, period_in_phi,
                                                  target_num_circles_phi)[1]
    return p, increase_in_p

def compare_directories(dir1, dir2) -> tuple:
    """Compare the contents of two directories, assuming no subdirectories. Return True/False and a brief explanation.

    Compare the names and the contents of the files, but not other metadata.
    Used for testing purposes.
    """
    comparison = filecmp.dircmp(dir1, dir2)
    if comparison.left_only:
        return False, "left only"
    if comparison.right_only:
        return False, "right only"
    for file_name in comparison.common_files:
        file1 = os.path.join(dir1, file_name)
        file2 = os.path.join(dir2, file_name)
        if not filecmp.cmp(file1, file2, shallow=False):
            return False, f"difference in {file_name}"
    return True, ""


def cross2d(x, y):
    """The cross-product of two plain vectors.

    Copied from https://numpy.org/doc/stable/reference/generated/numpy.cross.html"""
    return x[0] * y[1] - x[1] * y[0]


class Angle:
    """Defines an angle on the plane.

    :param angle_vertex: array-like containing the two coordinates of the angle vertex.
    :param angle_side_vector0: array-like containing the two coordinates of the vector whose direction specifies the
        0th side of the angle.
    :param angle_side_vector1: array-like containing the two coordinates of the vector whose direction specifies the
        1st side of the angle.
    """

    def __init__(self, angle_vertex: tuple, angle_side_vector0: tuple, angle_side_vector1: tuple):
        cross = cross2d(angle_side_vector0, angle_side_vector1)
        if cross == 0:
            raise ValueError(f"Sides of the angle are collinear: {angle_side_vector0, angle_side_vector1}")
        self.angle_side_vector0, self.angle_side_vector1 = (angle_side_vector0, angle_side_vector1) if cross > 0 else (
            angle_side_vector1, angle_side_vector0)
        self.scalar0 = cross2d(self.angle_side_vector0, angle_vertex)
        self.scalar1 = cross2d(angle_vertex, self.angle_side_vector1)

    def check_plain_point_inside_angle(self, point: tuple) -> bool:
        """Checks whether a plain point lies inside an angle.

        Assumes (and validates) that angle_side_vector0, angle_side_vector1 are not collinear.
        :param point: array-like containing the two coordinates of the point to be checked.
        :returns True if point lies inside the angle or on its boundary.
        """
        return cross2d(self.angle_side_vector0, point) >= self.scalar0 and cross2d(point, self.angle_side_vector1) >= self.scalar1


class HalfPlane:
    def __init__(self, point_on_boundary: tuple, line_direction_vector: tuple, point_in_other_half: tuple, precision):
        """Defines a half-plane.

        :param point_on_boundary: array-like containing the two coordinates of a point on the boundary of the half-plane.
        :param line_direction_vector: array-like containing the two coordinates of the vector along the boundary
            of the half-plane.
        :param point_in_other_half: array-like containing the two coordinates of a point strictly outside the half-plane.
        :param precision: a value (usually small) that extends the half-plane if positive, and reduces it otherwise:
            is subtracted from the scalar of the linear equation.  If line_direction_vector is a unit vector, indicates
            the distance of the extension.
        """
        self.normal = np.array((-line_direction_vector[1], line_direction_vector[0]))
        point_on_boundary = np.array(point_on_boundary)
        t = self.normal @ (np.array(point_in_other_half) - point_on_boundary)
        if t == 0:
            raise ValueError(f"The point that is supposed to define side of the boundary of the half-plane,"
                             f" lies on the boundary.  Direction of the boundary: {line_direction_vector},"
                             f"point on the boundary: {point_on_boundary},"
                             f" point that should specify the side: {point_in_other_half}")
        if t > 0:
            self.normal = -self.normal
        self.scalar = self.normal @ point_on_boundary - precision

    def check_plain_point_in_half_plane(self, point: tuple) -> bool:
        """Checks whether a point lies in the half-plane.

        :param point: array-like containing the two coordinates of the point to be checked.
        :returns True if point is inside or on the boundary of the half-plane.
        """
        return self.normal @ point >= self.scalar


class _ConditionFindingSeparatrix:
    """The class that defines the condition function for finding separatrix.

    (For using condition functions, see iterate_function_until_condition(..) above.)
    :param singular_point: array-like (2,), coordinates of the singular (periodic) point for which we are finding
        the separatrix.
    :param approach_separatrix_direction: array-like (2,), the coordinates of the eigenvector of the differential
        of the map that returns to the singular point; this is the eigenvector that indicates the direction from
        which we are approaching the singular point to find the separatrix.
    :param other_separatrix_direction: array-like (2,), the coordinates of the eigenvector of the differential
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
    """
    def __init__(self, singular_point, approach_separatrix_direction, other_separatrix_direction, precision,
                 separating_coefficient):
        self.singular_point = np.array(singular_point)
        approach_separatrix_direction = np.array(approach_separatrix_direction)
        self.approach_separatrix_direction = approach_separatrix_direction / np.linalg.norm(
            approach_separatrix_direction)
        other_separatrix_direction = np.array(other_separatrix_direction)
        self.other_separatrix_direction = other_separatrix_direction /np.linalg.norm(other_separatrix_direction)
        positive_bound_forward_direction = (separating_coefficient * self.approach_separatrix_direction
                                            + self.other_separatrix_direction)
        positive_bound_forward_direction = positive_bound_forward_direction / np.linalg.norm(positive_bound_forward_direction)
        positive_bound_backward_direction = (-separating_coefficient * self.approach_separatrix_direction
                                             + self.other_separatrix_direction)
        positive_bound_backward_direction = positive_bound_backward_direction / np.linalg.norm(positive_bound_backward_direction)
        self.positive_angle = Angle(self.singular_point, positive_bound_forward_direction, positive_bound_backward_direction)
        negative_bound_forward_direction = (separating_coefficient * self.approach_separatrix_direction
                                            - self.other_separatrix_direction)
        negative_bound_forward_direction = negative_bound_forward_direction / np.linalg.norm(negative_bound_forward_direction)
        negative_bound_backward_direction = (-separating_coefficient * self.approach_separatrix_direction
                                             - self.other_separatrix_direction)
        negative_bound_backward_direction = negative_bound_backward_direction / np.linalg.norm(negative_bound_backward_direction)
        self.negative_angle = Angle(self.singular_point, negative_bound_forward_direction, negative_bound_backward_direction)
        self.back_angle = Angle(self.singular_point, positive_bound_backward_direction, negative_bound_backward_direction)
        self.precision = precision

    def condition_fn_for_separatrix(self, point):
        """The condition function for finding the separatrix.

        :param point: array-like (2,), coordinates of the point to be checked.
        :returns 0, if the point is close to the singular one or is in the back angle (the latter possibility normally
            should not happen), 1 if it is within the positive angle, -1 if it is within the negative angle,
            None otherwise.
        """
        point = np.array(point)
        if np.linalg.norm(point - self.singular_point) <= self.precision:
            return 0
        if self.back_angle.check_plain_point_inside_angle(point):
            return 0
        if self.positive_angle.check_plain_point_inside_angle(point):
            return 1
        if self.negative_angle.check_plain_point_inside_angle(point):
            return -1
        return None


class InitialPointsForFindingSeparatrix:
    """Algorithm for starting points for binary search for finding initial point of an orbit in separatrix.

    Each next starting point is closer to the singular point from the previous one,
    singular_point * (1 - approaching_ratio) + approaching_ratio * previous_starting_point.
    """
    def __init__(self, singular_point: np.array, approaching_ratio: np.array):
        self.approaching_ratio = approaching_ratio
        self.second_addend = (1 - approaching_ratio) * singular_point

    def next_starting_point_couple(self, starting_point_couple: tuple, _):
        """Takes a pair of starting points and returns the next pair."""
        return (self.approaching_ratio * starting_point_couple[0] + self.second_addend,
                self.approaching_ratio * starting_point_couple[1] + self.second_addend)

    def starting_points(self, initial_starting_point1: np.array, initial_starting_point2: np.array, num_pairs: int):
        """Returns an iterator creating pairs of starting points."""
        return itertools.accumulate(itertools.repeat(None, num_pairs), func=self.next_starting_point_couple,
                                    initial=(initial_starting_point1, initial_starting_point2))


class SearcherForSeparatrix:
    """Searching for a separatrix orbit that hits a hyperbolic point.

    :param billiard: to define the billiard map.
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
    """
    def __init__(self, billiard: AbstractBilliard, period: int, iteration_limit: int,
        hyperbolic_point, approach_separatrix_direction, other_separatrix_direction, precision, separating_coefficient,
        forward: bool):
        self.ending_condition = _ConditionFindingSeparatrix(
            hyperbolic_point, approach_separatrix_direction, other_separatrix_direction, precision,
            separating_coefficient)
        self.billiard = billiard
        self.period = period
        self.iteration_limit = iteration_limit
        self.forward = forward

    def _iteration_towards_hyperbolic(self, previous_orbit: np.array) -> np.array:
        # apply period iterations starting from the last point in previous_orbit, and drop the starting point
        # which is already contained in previous_orbit:
        next_orbit = calculate_orbit(
            self.billiard, initial_phi=previous_orbit[-1, 1], initial_p=previous_orbit[-1, 2],
            num_iterations=self.period, start_numbering=previous_orbit[-1, 0], forward=self.forward)[1:, :]
        return self.billiard.set_orbit_around_point(next_orbit, self.ending_condition.singular_point)

    behind = "behind"  # the case when the orbit towards the hyperbolic point is discovered to be behind it;
    # this indicates that either separating_coefficient is too small, or the starting point is too far from
    # the hyperbolic point.

    def orbit_towards_hyperbolic_point(self, starting_point) -> tuple:
        """Run an orbit towards a hyperbolic point until it gets clear if it has been hit or missed.

        :param starting_point: (phi, p) array-like to start the orbit.
        :returns result_code, orbit
            result_code is  1 if the orbit went in the direction of other_separatrix_direction,
                            -1 if the orbit went in the direction opposite to other_separatrix_direction,
                            0 if the orbit got close to hyperbolic_point within precision,
                            "behind" if the orbit got into the back angle without getting close to hyperbolic_point,
                            None if iteration_limit was insufficient to get to any of the above.
            orbit contains all the intermediate billiard points before getting to any of the above result
                (maximum length period * iteration_limit + 1 points).
        """
        initial_orbit = np.array([[0, starting_point[0], starting_point[1]]])  # shape = (1, 3)
        exit_code, orbit_pieces = iterate_function_until_condition(
            self._iteration_towards_hyperbolic, initial_orbit,
            condition_fn=lambda orbit: self.ending_condition.condition_fn_for_separatrix(orbit[-1, 1:]),
            num_iter=self.iteration_limit)
        if exit_code == 0:  # check whether we are indeed close to hyperbolic_point, or are behind it:
            last_point = orbit_pieces[-1][-1, 1:]
            if np.linalg.norm(last_point - self.ending_condition.singular_point) > self.ending_condition.precision:
                exit_code = SearcherForSeparatrix.behind
        # merge all the arrays:
        orbit = np.concatenate(orbit_pieces)
        return exit_code, orbit

    def orbit_hit_hyperbolic_point(self, starting_point1, starting_point2, verbose=False) -> np.array:
        """Find a separatrix orbit that hits a hyperbolic point, by searching a given interval for a starting point.

        Uses binary search in the straight line interval [starting_point1, starting_point2].
        :param starting_point1: (phi, p) array-like to start the orbit, one of the bounds for the plane segment to search
            for the starting point to hit the hyperbolic point within precision.
        :param starting_point2: (phi, p) array-like to start the orbit, the other bound for the plane segment to search
            for the starting point to hit the hyperbolic point within precision.
        :param verbose: if True, print out the progress.
        :returns orbit, a np.array containing all the intermediate billiard points before getting to the hyperbolic point
            within precision (maximum length period * iteration_limit + 1 points).
        :raises ValueError if the orbits that start from starting_point1 and starting_point2, pass at the same side of
            hyperbolic point.
        :raises RuntimeError if an orbit that starts at one of the binary search points
            (including starting_point1, starting_point2), passes behind the hyperbolic point, or
            if binary search get to the current mpmath precision limit without the result (this may happen either if
            the separatrix crosses the angle defined by separating_coefficient - in which case starting points need
            to be chosen closer to the hyperbolic point - or in the unlikely case that mp.mp.dps is insufficient).
        """
        starting_point1 = np.array(starting_point1)
        starting_point2 = np.array(starting_point2)
        result_code1, orbit = self.orbit_towards_hyperbolic_point(starting_point1)
        if result_code1 == 0:
            return orbit
        result_code2, orbit = self.orbit_towards_hyperbolic_point(starting_point2)
        if result_code2 == 0:
            return orbit
        if result_code1 == SearcherForSeparatrix.behind or result_code2 == SearcherForSeparatrix.behind:
            raise RuntimeError("An orbit turned out to be behind the hyperbolic point.")
        if result_code1 == result_code2:
            raise ValueError(f"The orbits that start at the given starting points, pass at the same side"
                             f" of the hyperbolic: {result_code1}")
        while True:
            new_starting_point = (starting_point1 + starting_point2) / 2
            if np.array_equal(new_starting_point, starting_point1) or np.array_equal(
                    new_starting_point, starting_point2):
                raise RuntimeError(f"Binary search ended at the current dps = {mp.mp.dps} at\n"
                                   f"phi1, p1 = {starting_point1}, {result_code1}\n"
                                   f"phi2, p2 = {starting_point2}, {result_code2}\n")
            result_code_new, orbit = self.orbit_towards_hyperbolic_point(new_starting_point)
            if result_code_new == 0:
                return orbit
            if result_code_new == SearcherForSeparatrix.behind:
                raise RuntimeError("An orbit turned out to be behind the hyperbolic point.")
            if result_code_new == result_code1:
                starting_point1 = new_starting_point
            else:
                starting_point2 = new_starting_point
            if verbose:
                print(f"point1: {starting_point1}\npoint2: {starting_point2}\n")
                print(f"rel distance phi: "
                      f"{(starting_point2[0] - starting_point1[0]) / max(starting_point1[0], starting_point2[0])}")
                print(f"rel distance p: "
                      f"{(starting_point2[1] - starting_point1[1]) / max(starting_point1[1], starting_point2[1])}")

    def fill_separatrix(self, existing_separatrix: np.ndarray, starting_point1, starting_point2,
                        approaching_ratio: float, num_orbits: int, num_steps_back: int, verbose: bool = False) -> tuple:
        """Fill a separtrix with orbits that hit the hyperbolic point; extend the existing array of orbits.

        :param existing_separatrix: a np.array of orbits, shape = (orbit_length, 3, num_exising_orbits), or None.
        :param starting_point1: To find one orbit using orbit_hit_hyperbolic_point(.).
        :param starting_point2: To find one orbit using orbit_hit_hyperbolic_point(.).
        :param approaching_ratio: Each next orbit is built from starting_point1 and starting_point2 obtained
            from the previous values by multiplying their vectors from the hyperbolic_point by approaching_ratio.
        :param num_orbits: The number of the new orbits to create.
        :param num_steps_back: The number of steps away from the hyperbolic point to extend each of the new orbits.
            Do not extend if num_steps <= 0.
        :param verbose: reports details in finding the starting point of each separatrix orbit;
            helpful if the eigenvalues are close to 1.
        :return: separatrix, next_starting_point1, next_starting_point2
            Here separatrix is a np.array of orbits, shape = (orbit_length, 3, num_exising_orbits + num_orbits),
                which extends existing_separatrix if the latter is not None.
                In case some of the orbits are shorter than others, they are padded at the end with their last points.
            next_starting_point1, next_starting_point2 are the values of starting_point1, starting_point2 to be used
                to continue building separatrix orbits beyond num_orbits.
        :raises ValueError if the orbits that start from starting_point1 and starting_point2 at any step,
            pass at the same side of hyperbolic point.
        :raises RuntimeError if an orbit that starts at one of the binary search points
            (including starting_point1, starting_point2), passes behind the hyperbolic point, or
            if binary search get to the current mpmath precision limit without the result (this may happen either if
            the separatrix crosses the angle defined by separating_coefficient - in which case starting points need
            to be chosen closer to the hyperbolic point - or in the unlikely case that mp.mp.dps is insufficient).
        """
        # set the working precision:
        self.billiard.set_working_precision()
        # create the list of couples of starting points using itertools instead of a loop:
        starting_points = list(InitialPointsForFindingSeparatrix(
            self.ending_condition.singular_point, approaching_ratio).starting_points(
            starting_point1, starting_point2, num_orbits))
        next_starting_point1, next_starting_point2 = starting_points[-1]
        starting_points = starting_points[:-1]
        orbits = [self.orbit_hit_hyperbolic_point(*starting_point_couple, verbose)
                  for starting_point_couple in starting_points]
        # pad shorter orbits:
        max_orbit_length = max([orbit.shape[0] for orbit in orbits])
        if existing_separatrix is not None:
            max_orbit_length = max(max_orbit_length, existing_separatrix.shape[0] - max(num_steps_back, 0))
        orbits = [np.pad(orbit, ((0, max_orbit_length - orbit.shape[0]), (0, 0)), "edge")
                  for orbit in orbits]
        orbits_array = np.stack(orbits, axis=-1)
        orbits_array = extend_separatrix(self.billiard, self.forward, orbits_array, num_steps_back)
        max_orbit_length += max(num_steps_back, 0)
        if existing_separatrix is None:
            return orbits_array, next_starting_point1, next_starting_point2
        existing_separatrix = np.pad(
            existing_separatrix, ((0, max_orbit_length - existing_separatrix.shape[0]), (0, 0), (0, 0)),
            "edge")
        return np.concat((existing_separatrix, orbits_array), axis=-1), next_starting_point1, next_starting_point2


def extend_separatrix(billiard: AbstractBilliard, forward: bool, separatrix_to_extend: np.ndarray,
                      num_steps: int) -> np.ndarray:
    """Extend each of the orbits comprising separatrix_to_extend, away from the hyperbolic point.

    :param billiard: to define the billiard map.
    :param forward: if False, run the orbit backwards w.r. to the billiard.  (This is necessary if the eigenvalue of
        the differential in approach_separatrix_direction is > 1: then the forward direction of the billiard
        is away from the hyperbolic_point.)
    :param separatrix_to_extend: a np.array of orbits, shape = (orbit_length, 3, num_orbits).
    :param num_steps: The number of steps away from the hyperbolic point to extend each of the orbits comprising
        separatrix_to_extend.  Do not extend if num_steps <= 0.
    :return: a np.array of orbits, shape = (orbit_length + num_steps, 3, num_orbits), if num_steps > 0,
        separatrix_to_extend otherwise.
    """
    if num_steps <= 0:
        return separatrix_to_extend

    def extension(start: np.ndarray) -> np.ndarray:
        """Given a 1-dim array [number, phi, p], creates an orbit starting from it."""
        return calculate_orbit(billiard, start[1], start[2], num_steps, start[0], not forward)

    extension_to_separatrix = np.apply_along_axis(extension, 0, separatrix_to_extend[0, :, :])
    # change the direction of numbering within the orbits:
    extension_to_separatrix[:, 0, :] = 2 * extension_to_separatrix[0, 0, :] - extension_to_separatrix[:, 0, :]
    # flip the order of points in the orbits:
    extension_to_separatrix = np.flip(extension_to_separatrix, 0)
    # concatenate the extension with the original, dropping the overlapping points which are
    # the last in the extension and the 0th in the original:
    return np.concat((extension_to_separatrix[:-1, :, :], separatrix_to_extend))


def create_validate_separatrix_by_blocks(
        billiard_main: AbstractBilliard, billiard_val: AbstractBilliard, period: int, iteration_limit: int,
        hyperbolic_point, approach_separatrix_direction, other_separatrix_direction, precision, separating_coefficient,
        forward: bool, starting_point1, starting_point2,
        approaching_ratio: float, num_new_orbits: int, block_size: int, num_steps_back: int, filepaths: dict,
        verbose: bool = False) -> float:
    """Create, validate and write to disk a separatrix by blocks of orbits of specified size.

    :param billiard_main: an instance of AbstractBilliard representing the boundary with the precision
        to be used to calculate the output orbit.
    :param billiard_val: an instance of AbstractBilliard representing the boundary with the validation precision
        (which is assumed to be greater than the main working precision).
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
    :param filepaths: a dict of filepaths with the following keys:
        "separatrix": contains the current version of the separatrix.
        "temp_separatrix": contains the temp version of the separatrix, before being renamed into the final version.
        "misc_outputs": a pkl file containing the table of various outputs including validation discrepancies, a table
            with one row for each new block of orbits created, standard index and the following columns:
                start_block: the initial orbit number in the block.
                starting_point1, starting_point2: as used to create the orbits in the block.
                discrepancy: between the main and the val orbit, as returned by compare_orbits(.).
                time_finished: The timestamp at which the calculations were finished.
        "temp_misc_outputs": the temp version of misc_outputs, before being renamed into the final version.
    :param verbose: reports details in finding the starting point of each separatrix orbit;
        helpful if the eigenvalues are close to 1.
    :returns max discrepancy.
    :raises ValueError if the orbits that start from starting_point1 and starting_point2 at any step,
        pass at the same side of hyperbolic point.
    :raises RuntimeError if an orbit that starts at one of the binary search points
        (including starting_point1, starting_point2), passes behind the hyperbolic point, or
        if binary search get to the current mpmath precision limit without the result (this may happen either if
        the separatrix crosses the angle defined by separating_coefficient - in which case starting points need
        to be chosen closer to the hyperbolic point - or in the unlikely case that mp.mp.dps is insufficient).
    """
    searcher_main = SearcherForSeparatrix(
        billiard_main, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
        other_separatrix_direction, precision, separating_coefficient, forward)
    searcher_val = SearcherForSeparatrix(
        billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
        other_separatrix_direction, precision, separating_coefficient, forward)
    # Get the current state on disk:
    existing_separatrix = None
    separatrix_file_exists = os.path.exists(filepaths["separatrix"])
    temp_separatrix_file_exists = os.path.exists(filepaths["temp_separatrix"])
    if separatrix_file_exists:
        existing_separatrix = unpickle(filepaths["separatrix"])
        if temp_separatrix_file_exists:
            os.remove(filepaths["temp_separatrix"])
    elif temp_separatrix_file_exists:
        existing_separatrix = unpickle(filepaths["temp_separatrix"])
        os.replace(filepaths["temp_separatrix"], filepaths["separatrix"])
    misc_outputs = None
    misc_outputs_file_exists = os.path.exists(filepaths["misc_outputs"])
    temp_misc_outputs_file_exists = os.path.exists(filepaths["temp_misc_outputs"])
    if misc_outputs_file_exists:
        misc_outputs = unpickle(filepaths["misc_outputs"])
        if temp_misc_outputs_file_exists:
            os.remove(filepaths["temp_misc_outputs"])
    elif temp_misc_outputs_file_exists:
        misc_outputs = unpickle(filepaths["temp_misc_outputs"])
        os.replace(filepaths["temp_misc_outputs"], filepaths["misc_outputs"])
    # If misc_outputs exist, the next block data are already in it in the last row:
    start_block = 0
    if misc_outputs is not None:
        if len(misc_outputs) == 0:
            raise RuntimeError("misc_outputs file exists but is an empty table")
        starting_point1 = misc_outputs.loc[len(misc_outputs) - 1, "starting_point1"]
        starting_point2 = misc_outputs.loc[len(misc_outputs) - 1, "starting_point2"]
        start_block = misc_outputs.loc[len(misc_outputs) - 1, "start_block"]  # if the function has completed already,
        # start_block >= num_new_orbits and the following loop will not be executed.
    while start_block < num_new_orbits:
        separatrix_main, new_starting_point1, new_starting_point2 = searcher_main.fill_separatrix(
            existing_separatrix, starting_point1, starting_point2, approaching_ratio, block_size, num_steps_back,
            verbose)
        separatrix_val, _, _ = searcher_val.fill_separatrix(
            existing_separatrix, starting_point1, starting_point2, approaching_ratio, block_size, num_steps_back,
            verbose)
        # The following line is inefficient, as it compares already the full new separatrices rather than the new orbits
        # only.  Likely this inefficiency is small in comparison with the previous lines:
        discrepancy = float(billiard_main.compare_orbits(separatrix_main, separatrix_val))
        new_start_block = start_block + block_size
        if misc_outputs is None:
            misc_outputs = pd.DataFrame({
                "start_block": [start_block, new_start_block],
                "starting_point1": [starting_point1, new_starting_point1],
                "starting_point2": [starting_point2, new_starting_point2],
                "discrepancy": [discrepancy, None], "time_finished": [str(datetime.datetime.now()), None]})
        else:
            misc_outputs.loc[len(misc_outputs) - 1, "discrepancy"] = discrepancy
            misc_outputs.loc[len(misc_outputs) - 1, "time_finished"] = str(datetime.datetime.now())
            # create the row for the next block; at the end, keep it for the case
            # the function is called again; this row allows to continue for more orbits, or
            # helps to find that this loop has completed.
            misc_outputs = pd.concat(
                [misc_outputs, pd.DataFrame({"start_block": [new_start_block],
                                             "starting_point1": [new_starting_point1],
                                             "starting_point2": [new_starting_point2]})],
                ignore_index=True)
        pickle_one_object(misc_outputs, filepaths["temp_misc_outputs"])
        pickle_one_object(separatrix_main , filepaths["temp_separatrix"])
        os.replace(filepaths["temp_misc_outputs"], filepaths["misc_outputs"])
        os.replace(filepaths["temp_separatrix"], filepaths["separatrix"])
        start_block, starting_point1, starting_point2, existing_separatrix = (
            new_start_block, new_starting_point1, new_starting_point2, separatrix_main)
        if verbose:
            print(f"Written partial separatrix to {filepaths['separatrix']}")
    max_discrepancy = misc_outputs.discrepancy.max()
    print(f"Max discrepancy: {max_discrepancy}", flush=True)
    return max_discrepancy


def extend_validate_separatrix(billiard_main: AbstractBilliard, billiard_val: AbstractBilliard, forward: bool,
                               separatrix_to_extend: np.ndarray, num_steps: int) -> tuple:
    """Extend each of the orbits comprising separatrix_to_extend, away from the hyperbolic point, and validate.

    :param billiard_main: an instance of AbstractBilliard representing the boundary with the precision
        to be used to calculate the output orbit.
    :param billiard_val: an instance of AbstractBilliard representing the boundary with the validation precision
        (which is assumed to be greater than the main working precision).
    :param forward: if False, run the orbit backwards w.r. to the billiard.  (This is necessary if the eigenvalue of
        the differential in approach_separatrix_direction is > 1: then the forward direction of the billiard
        is away from the hyperbolic_point.)
    :param separatrix_to_extend: a np.array of orbits, shape = (orbit_length, 3, num_orbits).
    :param num_steps: The number of steps away from the hyperbolic point to extend each of the orbits comprising
        separatrix_to_extend.  Do not extend if num_steps <= 0.
    :return: a tuple:
        a np.array of orbits, shape = (orbit_length + num_steps, 3, num_orbits), if num_steps > 0,
            separatrix_to_extend otherwise.
        max_discrepancy between the calculation with billiard_main and billiard_val, float.
    """
    result = extend_separatrix(billiard_main, forward, separatrix_to_extend, num_steps)
    result_val = extend_separatrix(billiard_val, forward, separatrix_to_extend, num_steps)
    max_discrepancy = float(np.max(np.abs(result_val - result)))
    return result, max_discrepancy


def extend_validate_separatrix_from_file(billiard_main: AbstractBilliard, billiard_val: AbstractBilliard, forward: bool,
                                         separatrix_filepath, num_steps: int, log_filepath=None,
                                         new_separatrix_filepath=None) -> float:
    """Extend each of the orbits comprising the given separatrix, away from the hyperbolic point, and validate.

    :param billiard_main: an instance of AbstractBilliard representing the boundary with the precision
        to be used to calculate the output orbit.
    :param billiard_val: an instance of AbstractBilliard representing the boundary with the validation precision
        (which is assumed to be greater than the main working precision).
    :param forward: if False, run the orbit backwards w.r. to the billiard.  (This is necessary if the eigenvalue of
        the differential in approach_separatrix_direction is > 1: then the forward direction of the billiard
        is away from the hyperbolic_point.)
    :param separatrix_filepath: the path to the file containing the separatrix to extend.
    :param num_steps: The number of steps away from the hyperbolic point to extend each of the orbits comprising
        separatrix_to_extend.  Do not extend if num_steps <= 0.
    :param log_filepath: if not None, a log message is appended to it.
    :param new_separatrix_filepath: the path to which the new separatrix is written; if None, written to
        separatrix_filepath.
    :return: max_discrepancy between the calculation with billiard_main and billiard_val, float.
    """
    separatrix_to_extend = unpickle(separatrix_filepath)
    new_separatrix, max_discrepancy = extend_validate_separatrix(
        billiard_main, billiard_val, forward, separatrix_to_extend, num_steps)
    if new_separatrix_filepath is None:
        new_separatrix_filepath = separatrix_filepath
    pickle_one_object(new_separatrix, new_separatrix_filepath)
    if log_filepath is not None:
        with open(log_filepath, "a") as file:
            file.write(f"\nExtended the separatrix away from the hyperbolic additional {num_steps} steps,"
                       f" max_discrepancy: {max_discrepancy}.")
    return max_discrepancy


def remove_front_back_of_orbits(orbit: np.array, front=None, back=None) -> np.array:
    """Remove front and/or back part of an orbit/separatrix.

    :param orbit: An orbit or an array of orbits (e.g., a separatrix), with 0th and 1st coordinates for each orbit
        being as created by calculate_orbit(.) and 2nd and further coordinates enumerating orbits, as follows:
        0th coordinate indicates points in one orbit, 1st coordinate takes the following values:
        0 for point number (it may be positive or negative in case of extending a separatrix back),
        1 for phi coordinate, 2 for p coordinate of the point.
        (Assumes that the point numbers are synchronized with the 0th coordinate, with the difference being constant
        for all orbits.)
    :param front: either None or int; in the latter case keep only the front part of orbit
        (the one that approaches the hyperbolic point as built) and remove all points with
        the point number < front.
    :param back: either None or int; in the latter case keep only the back part of orbit
        (removing the one that approaches the hyperbolic point as built); remove from all points with
        the point number > back.
    :return: orbit with removed front and/or back.
    """
    if front is not None:
        indicator_to_keep = orbit[:, 0, ...] >= front
        indices_to_keep = np.argwhere(indicator_to_keep)
        if indices_to_keep.size > 0:
            cutoff_index = indices_to_keep[:, 0].min()  # if point numbers are not synced with the 0th coord, keep max
            orbit = orbit[cutoff_index:,...]
        else:
            orbit = orbit[:0, ...]
    if back is not None:
        indicator_to_keep = orbit[:, 0, ...] <= back
        indices_to_keep = np.argwhere(indicator_to_keep)
        if indices_to_keep.size > 0:
            cutoff_index = indices_to_keep[:, 0].max()  # if point numbers are not synced with the 0th coord, keep max
            orbit = orbit[:cutoff_index + 1,...]
        else:
            orbit = orbit[:0, ...]
    return orbit


def flip_phi_orbit(orbit: np.array, flip_phi, period_phi) -> np.array:
    """Flip an orbit w.r. to a given value of phi.

    :param orbit: An orbit or an array of orbits (e.g., a separatrix), with 0th and 1st coordinates for each orbit
        being as created by calculate_orbit(.) and 2nd and further coordinates enumerating orbits, as follows:
        0th coordinate indicates points in one orbit, 1st coordinate takes the following values:
        0 for point number (it may be positive or negative in case of extending a separatrix back),
        1 for phi coordinate, 2 for p coordinate of the point.
        (Assumes that the point numbers are synchronized with the 0th coordinate, with the difference being constant
        for all orbits.)
        The original orbit is also flipped.
    :param flip_phi: if not None, flip the orbit w.r. to the line phi = flip_phi.  If None, return orbit unchanged.
    :param period_phi: if not None, take the residue of all phi values w.r. to period_phi.
    :return: the flipped orbit, for convenience (the original orbit is also changed).
    """
    if flip_phi is None:
        return orbit
    orbit[:, 1, ...] = 2 * flip_phi - orbit[:, 1, ...]
    if period_phi is not None:
        orbit[:, 1, ...] = orbit[:, 1, ...] % period_phi
    return orbit
