# based on https://github.com/uncertainty-toolbox/uncertainty-toolbox

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Any, NoReturn

Numeric = Union[int, float, np.ndarray]

def plot_xy_specifyBound(
    y_pred: np.ndarray,

    y_UP: np.ndarray,
    y_LO: np.ndarray,

    y_true: np.ndarray,
    x: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    leg_loc: Union[int, str] = 3,
    ax: Union[matplotlib.axes.Axes, None] = None,
    interval_coverage = "95\%",
    title = None,
    x_label = "$x$",
    y_label = "$y$",
    dashStyle = False,
    legend = True
) -> matplotlib.axes.Axes:
    
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing x
    order = np.argsort(x)


    y_pred, y_UP, y_LO, y_true, x = (
        y_pred[order],
        y_UP[order],
        y_LO[order],
        y_true[order],
        x[order],
    )

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_UP, y_LO, y_true, x] = filter_subset([y_pred, y_UP, y_LO, y_true, x], n_subset)


    h1 = ax.plot(x, y_true, ".", mec="#ff7f0e", mfc="None")
    if dashStyle:
        h2 = ax.plot(x, y_pred, "--", c="#1f77b4", linewidth=2)
    else:
        h2 = ax.plot(x, y_pred, "-", c="#1f77b4", linewidth=2)
    h3 = ax.fill_between(
        x,
        y_LO,
        y_UP,
        color="lightsteelblue",
        alpha=0.4,
    )

    if legend:
        ax.legend(
            [h1[0], h2[0], h3],
            ["Observations", "Predictions", interval_coverage + " Interval"],
            loc=leg_loc,
        )

    # Format plot
    if ylims is not None:
        ax.set_ylim(ylims)

    if xlims is not None:
        ax.set_xlim(xlims)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Confidence Band")
    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return ax




def filter_subset(input_list: List[List[Any]], n_subset: int) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int
    n_total = len(input_list[0])


    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list




def plot_calibration(
    n_subset: Union[int, None] = None,
    curve_label: Union[str, None] = None,
    exp_props: Union[np.ndarray, None] = None,
    obs_props: Union[np.ndarray, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
    title = "Average Calibration"
) -> matplotlib.axes.Axes:
    
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    
    exp_proportions = np.array(exp_props).flatten()
    obs_proportions = np.array(obs_props).flatten()
    if exp_proportions.shape != obs_proportions.shape:
        raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set label
    if curve_label is None:
        curve_label = "Predictor"

    # Plot
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(exp_proportions, obs_proportions, label=curve_label, c="#1f77b4")
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Percentage Below")
    ax.set_ylabel("Observed Percentage Below")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title(title)

    # Compute miscalibration area
    miscalibration_area = miscalibration_area_from_proportions(
        exp_proportions=exp_proportions, obs_proportions=obs_proportions
    )

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )

    return ax


def miscalibration_area_from_proportions(
    exp_proportions: np.ndarray, obs_proportions: np.ndarray
) -> float:
    """Miscalibration area from expected and observed proportions lists.
    This function returns the same output as `miscalibration_area` directly from a list
    of expected proportions (the proportion of data that you expect to observe within
    prediction intervals) and a list of observed proportions (the proportion data that
    you observe within prediction intervals).
    Args:
        exp_proportions: expected proportion of data within prediction intervals.
        obs_proportions: observed proportion of data within prediction intervals.
    Returns:
        A single scalar that contains the miscalibration area.
    """
    areas = trapezoid_area(
        exp_proportions[:-1],
        exp_proportions[:-1],
        obs_proportions[:-1],
        exp_proportions[1:],
        exp_proportions[1:],
        obs_proportions[1:],
        absolute=True,
    )
    return areas.sum()



def trapezoid_area(
    xl: np.ndarray,
    al: np.ndarray,
    bl: np.ndarray,
    xr: np.ndarray,
    ar: np.ndarray,
    br: np.ndarray,
    absolute: bool = True,
) -> Numeric:
    """
    Calculate the area of a vertical-sided trapezoid, formed connecting the following points:
        (xl, al) - (xl, bl) - (xr, br) - (xr, ar) - (xl, al)
    This function considers the case that the edges of the trapezoid might cross,
    and explicitly accounts for this.
    Args:
        xl: The x coordinate of the left-hand points of the trapezoid
        al: The y coordinate of the first left-hand point of the trapezoid
        bl: The y coordinate of the second left-hand point of the trapezoid
        xr: The x coordinate of the right-hand points of the trapezoid
        ar: The y coordinate of the first right-hand point of the trapezoid
        br: The y coordinate of the second right-hand point of the trapezoid
        absolute: Whether to calculate the absolute area, or allow a negative area (e.g. if a and b are swapped)
    Returns: The area of the given trapezoid.
    """

    # Differences
    dl = bl - al
    dr = br - ar

    # The ordering is the same for both iff they do not cross.
    cross = dl * dr < 0

    # Treat the degenerate case as a trapezoid
    cross = cross * (1 - ((dl == 0) * (dr == 0)))

    # trapezoid for non-crossing lines
    area_trapezoid = (xr - xl) * 0.5 * ((bl - al) + (br - ar))
    if absolute:
        area_trapezoid = np.abs(area_trapezoid)

    # Hourglass for crossing lines.
    # NaNs should only appear in the degenerate and parallel cases.
    # Those NaNs won't get through the final multiplication so it's ok.
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = intersection((xl, bl), (xr, br), (xl, al), (xr, ar))[0]
    tl_area = 0.5 * (bl - al) * (x_intersect - xl)
    tr_area = 0.5 * (br - ar) * (xr - x_intersect)
    if absolute:
        area_hourglass = np.abs(tl_area) + np.abs(tr_area)
    else:
        area_hourglass = tl_area + tr_area

    # The nan_to_num function allows us to do 0 * nan = 0
    return (1 - cross) * area_trapezoid + cross * np.nan_to_num(area_hourglass)


def intersection(
    p1: Tuple[Numeric, Numeric],
    p2: Tuple[Numeric, Numeric],
    p3: Tuple[Numeric, Numeric],
    p4: Tuple[Numeric, Numeric],
) -> Tuple[Numeric, Numeric]:
    """
    Calculate the intersection of two lines between four points, as defined in
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection.
    This is an array option and works can be used to calculate the intersections of
    entire arrays of points at the same time.
    Args:
        p1: The point (x1, y1), first point of Line 1
        p2: The point (x2, y2), second point of Line 1
        p3: The point (x3, y3), first point of Line 2
        p4: The point (x4, y4), second point of Line 2
    Returns: The point of intersection of the two lines, or (np.nan, np.nan) if the lines are parallel
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D

    return x, y