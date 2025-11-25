"""
Originally from https://github.com/aaryapatil/elicit-disk-copulas.

Edited by Simon Alinder.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import label
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

"""
Example run:
thin_thick_mask = copula_split_finder(
    data,
    "FE_H",
    "MG_FE",
    n_levels=80,
    divergence_percentile_level_lower=50,
    median_separation_multiplier=3,
    plotting_level=1,
    manual_point=((0.05, 0.75), (0.9, 0.57)),
)
"""

# %% Auxiliary functions


def coords_to_eqn(coords):
    """
    Get equation for a line from two coordinates.

    Fit an equation of the form y = k x + m to a line passing through two points.

    Parameters
    ----------
    coords : list
        A list containing two tuples with x- and y-coordinates for two points.

    Returns
    -------
    slope : float
        k.
    intercept : float
        m.

    """
    if coords[1][0] == coords[0][0]:
        msg = "Vertical line detected in coords_to_eqn."
        print(msg)
        return np.inf, np.nan
    slope = (coords[1][1] - coords[0][1]) / (coords[1][0] - coords[0][0])
    intercept = coords[0][1] - slope * coords[0][0]
    return (slope, intercept)


def line_selection_over(coords, dataset, xcol="Jphi", ycol="Jz"):
    """
    Create a mask selecting stars above a given line.

    Parameters
    ----------
    coords : list
        A list of tuples of the coordinates of the line.
    dataset : astropy.table.QTable
        The data to make the selection in.

    Returns
    -------
    mask : np.ndarray
        The mask, with bools indicating a stars membership over the line or not.

    """
    mask = np.zeros(len(dataset), bool)
    line_xy = ([xy[0] for xy in coords], [xy[1] for xy in coords])

    for i in range(len(coords) - 1):
        slope, intercept = coords_to_eqn([coords[i], coords[i + 1]])
        mask_part = (
            (dataset[xcol].value >= line_xy[0][i])
            & (dataset[xcol].value <= line_xy[0][i + 1])
            & (dataset[ycol].value > slope * dataset[xcol].value + intercept)
        )
        mask = mask | mask_part
    return mask


def line_selection_under(coords, dataset, xcol="Jphi", ycol="Jz"):
    """
    Create a mask selecting stars below a given line.

    Parameters
    ----------
    coords : list
        A list of tuples of the coordinates of the line.
    dataset : QTable
        The data to make the selection in.

    Returns
    -------
    mask : np.ndarray
        The mask, with bools indicating a stars membership under the line or not.

    """
    mask = np.zeros(len(dataset), bool)
    line_xy = ([xy[0] for xy in coords], [xy[1] for xy in coords])

    for i in range(len(coords) - 1):
        slope, intercept = coords_to_eqn([coords[i], coords[i + 1]])
        mask_part = (
            (dataset[xcol].value >= line_xy[0][i])
            & (dataset[xcol].value <= line_xy[0][i + 1])
            & (dataset[ycol].value < slope * dataset[xcol].value + intercept)
        )
        mask = mask | mask_part
    return mask


def divergence(grid, step):
    """
    Compute the divergence in a 2D vector field.

    Parameters
    ----------
    grid : sequence of 2D arraysTrue
        The x- and y-compnents of the vector field.
    step : float
        Distance between points in grid. Assumed to be square.

    Returns
    -------
    2D array
        The divergence of the grid.

    """
    return np.ufunc.reduce(
        np.add, [np.gradient(grid[i], step, axis=i) for i in range(len(grid))]
    )


def line_opens_positive(line):
    """
    Check if a given line "opens" along the positive z-axis.

    i.e. if it is shaped like a C if axis == 0 as this shape "opens" along the positive
    x-direction.

    Will only work for well-behaved lines!

    Parameters
    ----------
    line : array
        [N, 2] array of the line coordinates.

    Returns
    -------
    bool
        True if the line opens in the positive direction, else False.

    """
    axis_coords = line[:, 0]
    big_end = np.max((axis_coords[0], axis_coords[-1]))
    small_end = np.min((axis_coords[0], axis_coords[-1]))
    if axis_coords.min() < small_end and axis_coords.max() == big_end:
        return True
    if axis_coords.max() > big_end and axis_coords.min() == small_end:
        return False
    # Check if the line is a closed loop
    if all(line[0] == line[-1]):
        # Return True of the shape is enlongated along the considered axis
        return bool(
            (line[:, 0].max() - line[:, 0].min())
            > (line[:, 1].max() - line[:, 1].min())
        )
    print("Could not determine how one of the lines opens.")
    return np.nan


def compare_copula_plot(select_stars, ecdf_eval, xcol="FE_H", ycol="MG_FE"):
    """
    Make a plot comparing the given plane to the flat probability density plane of the same.

    Parameters
    ----------
    select_stars : QTable
        Table of stars to separate.
    ecdf_eval : list
        List of two ndarrays that are the ECDF evaluated on the data.
        Obtained by running the quick_copula function.
    xcol : str
        Name of column to use on x-axis.
    ycol : str
        Name of column to use on y-axis.

    Returns
    -------
    None. Makes a plot.

    """
    x_string = f"[{xcol.split("_")[0].capitalize()}/{xcol.split("_")[1].capitalize()}]"
    y_string = f"[{ycol.split("_")[0].capitalize()}/{ycol.split("_")[1].capitalize()}]"
    x_limits = (-1.3, 0.5)
    y_limits = (-0.1, 0.41)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplots_adjust(hspace=0, wspace=0.4)
    im = ax[0].hist2d(
        select_stars[xcol].value,
        select_stars[ycol].value,
        bins=180,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
        range=(x_limits, y_limits),
        rasterized=True,
    )[3]

    ax[0].set_xlim(x_limits)
    ax[0].set_ylim(y_limits)
    ax[0].set_xlabel(x_string + r"$\; \mathrm{or} \; x_1$")
    ax[0].set_ylabel(y_string + r"$\; \mathrm{or} \; x_2$")
    ax[0].set_xticks(np.linspace(x_limits[0], x_limits[1], num=5, endpoint=True))
    ax[0].set_yticks(np.linspace(y_limits[0], y_limits[1], num=5, endpoint=True))
    plt.colorbar(im, ax=ax[0], orientation="horizontal", label="Number Density")

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax[0])
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 0.8, pad=0.2, sharex=ax[0])
    ax_histy = divider.append_axes("right", 0.8, pad=0.2, sharey=ax[0])

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # Plot margins of data
    kdex = stats.gaussian_kde(select_stars[xcol])
    kdey = stats.gaussian_kde(select_stars[ycol])
    xx = np.linspace(*x_limits, 800)
    xy = np.linspace(*y_limits, 800)
    ax_histx.plot(xx, kdex(xx), color="black", lw=3, label=r"$f_1$")
    ax_histy.plot(kdey(xy), xy, color="black", lw=3, label=r"$f_2$")
    ax_histx.legend(fontsize=10)
    ax_histy.legend(fontsize=10)

    im = ax[1].hist2d(
        ecdf_eval[0],
        ecdf_eval[1],
        bins=180,
        norm=mpl.colors.LogNorm(),
        cmap="magma",
        rasterized=True,
    )[3]
    ax[1].set_xlim(-0.075, 1.05)
    ax[1].set_ylim(-0.075, 1.075)
    ax[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[1].set_xticklabels([0, "", 0.5, "", 1])
    ax[1].set_yticklabels([0, "", 0.5, "", 1])
    ax[1].set_xlabel(r"$u_1$")
    ax[1].set_ylabel(r"$u_2$")
    plt.colorbar(im, ax=ax[1], orientation="horizontal", label="Number Density")
    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax[1])
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 0.8, pad=0.2, sharex=ax[1])
    ax_histy = divider.append_axes("right", 0.8, pad=0.2, sharey=ax[1])

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # Plot margins of copula
    dat_range = np.linspace(-0.075, 1.075, 1000)
    ax_histx.plot(dat_range, stats.uniform.pdf(dat_range), color="black", lw=3)
    ax_histy.plot(stats.uniform.pdf(dat_range), dat_range, color="black", lw=3)
    ax_histx.set_ylim(0, 1.25)
    ax_histy.set_xlim(0, 1.25)

    fig.text(0.48, 0.55, r"$\rightarrow$", fontsize=30)
    fig.text(0.47, 0.65, r"$F_{1}(x_1)$", fontsize=20)
    fig.text(0.47, 0.6, r"$F_{2}(x_2)$", fontsize=20)
    plt.savefig("copula_transform.pdf")


def quick_copula(data):
    """
    Compute the EDF and KDE of the EDF.

    Parameters
    ----------
    data : list
        List containing the x and y columns of the data.

    Returns
    -------
    ecdf : list
        Empirical Distribution Function of the x and y columns of the data.
    ecdf_eval : list
        The Empirical Distribution Function of the x and y columns of the data
        evaluated on the data.
    u_grid1 : ndarray
        Output from np.meshgrid.
    u_grid2 : ndarray
        Output from np.meshgrid.
    kde_grid : ndarray
        Kernel Density Estimate of the EDF data.

    """
    ecdf = []
    ecdf_eval = []
    for arr in data:
        ecdf.append(ECDF(arr))
        ecdf_eval.append(ecdf[-1](arr))

    u_grid1, u_grid2 = np.meshgrid(
        np.linspace(0.001, 0.999, 50), np.linspace(0.001, 0.999, 50)
    )
    kernel = gaussian_kde(np.vstack(ecdf_eval))
    kde_grid = np.reshape(
        kernel(np.vstack([u_grid1.ravel(), u_grid2.ravel()])),
        u_grid1.shape,
    )

    return ecdf, ecdf_eval, u_grid1, u_grid2, kde_grid


# %% Main function
def copula_split_finder(
    select_stars,
    xcol="FE_H",
    ycol="MG_FE",
    n_levels=100,
    min_points_in_area=10,
    divergence_percentile_level_upper=100.0,
    divergence_percentile_level_lower=50.0,
    median_separation_multiplier=4.0,
    polydeg=3,
    plotting_level=0,
    manual_point=None,
):
    """
    Split a sample of stars by using a copula.

    Parameters
    ----------
    select_stars : astropy.table.QTable
        Table of stars to separate.
    xcol : str
        Name of column to use on x-axis.
    ycol : str
        Name of column to use on y-axis.
    n_levels : int, optional
        Number of contours to use. The default is 100.
    min_points_in_area : int, optional
        How many points need to be in the area for it to be considered. The default is 10.
    divergence_percentile_level_upper : float, optional
        The percentile to remove things greater than in the divergence map.
        Exists because there is noise around the edges. The default is 100.0.
    divergence_percentile_level_lower : float, optional
        The percentile to remove things lower than in the divergence map.
        Can be used to make low quality maps work better. The default is 50.0.
    median_separation_multiplier : float, optional
        The factor to multiply the median point-to-point distance which determines
        whether a line is continuous or has a jump in it.
        The default is 4.0.
    polydeg : int, optional
        Degree of the polynomial to fit to the points. Default is 3.
    plotting_level : int, optional
        How many plots should be shown.
        0 gives no plots at all. 1 gives the major results. 2 gives all plots including debugging.
        The default is 0.
    manual_point : tuple, optional
        2-tuple containing the x-y coordinates of a single point to add to the solution.
        Can also be a tuple of tuples of x-y coordinates of several points.

    Raises
    ------
    Warning
        Raised if there is an issue defining the area to search in.

    Returns
    -------
    above_inds : array
        Boolean indexing array for which rows in select_stars are above the copula.
    below_inds : array
        Boolean indexing array for which rows in select_stars are below the copula.
    det_fit_line : np.poly1d
        Numpy polynomial that describes the copula in the given space.

    """
    print("Computing copula")
    ecdf, ecdf_eval, u_grid1, u_grid2, kde_grid = quick_copula(
        [select_stars[xcol], select_stars[ycol]]
    )

    if plotting_level:
        compare_copula_plot(select_stars, ecdf_eval, xcol, ycol)

    # Grid plots
    # Automaded procedure. Written by Simon Alinder
    print("Make copula grid plots")

    # Compute copula contours
    plt.figure(figsize=(8, 6))
    cop_contour = plt.contour(u_grid1, u_grid2, kde_grid, levels=n_levels)

    # Compute flow line using contours
    # First and last elements are always empty
    vertices = [path.vertices for path in cop_contour.get_paths()[1:-1]]

    grad = np.gradient(kde_grid)
    div = divergence(grad, np.diff(u_grid1[0, :])[0])

    if plotting_level > 1:
        plt.figure(figsize=(8, 6))
        plt.quiver(u_grid1, u_grid2, grad[1], grad[0])
        plt.contour(u_grid1, u_grid2, kde_grid, levels=50)
        plt.title("Scalar field gradient")

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(
            u_grid1,
            u_grid2,
            div,
            shading="nearest",
            vmin=np.percentile(div, 1),
            vmax=np.percentile(div, 99),
        )
        plt.colorbar(extend="both")
        plt.quiver(u_grid1, u_grid2, grad[1], grad[0])
        plt.title("Vector field divergence")
        plt.savefig("vector_div.pdf")

    div_map = np.where(div > -1, div, 0.0)
    div_map = np.where(
        (np.percentile(div, divergence_percentile_level_lower) < div)
        & (div < np.percentile(div, divergence_percentile_level_upper)),
        div_map,
        0.0,
    )  # Because of noisy edges

    # We know we don't care about the lower areas of the space
    div_map = div_map[div_map.shape[0] // 2 :, :]
    area_shift = 0.5
    labeled_area, n_features = label(div_map)
    if plotting_level > 1:
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_area, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title("Areas identified")

    first_edge = labeled_area[0, :]
    second_edge = labeled_area[:, 0]
    area_ids_on_both_edges = np.unique(first_edge)[
        np.isin(np.unique(first_edge), np.unique(second_edge))
    ]
    area_on_both_edges = np.isin(
        np.unique(labeled_area), np.unique(first_edge)
    ) & np.isin(np.unique(labeled_area), np.unique(second_edge))

    if area_on_both_edges.sum() == 1:
        msg = "Only one area touches both sides. This is not what is expected."
        raise Warning(msg)

    if area_on_both_edges.sum() > 2:  # noqa: PLR2004
        # Use two methods to detect the right area. If they don't agree, raise a Warning
        # First method, amount of contact with edge.
        # First hit is 0, the background, discard it
        n_edge_pixels_x_axis = np.unique(first_edge, return_counts=True)[1][
            area_on_both_edges[np.unique(first_edge)]
        ][1:]
        n_edge_pixels_y_axis = np.unique(second_edge, return_counts=True)[1][
            area_on_both_edges[np.unique(second_edge)]
        ][1:]
        n_edge_pixels_total = np.sum(
            (n_edge_pixels_x_axis, n_edge_pixels_y_axis), axis=0
        )
        matches = (n_edge_pixels_total == n_edge_pixels_total.max()).nonzero()[0]
        index_of_larger = int(matches[0])
        edges_area_id = int(area_ids_on_both_edges[1 + index_of_larger])

        # Second method, area size.
        all_area_sizes = np.unique(labeled_area, return_counts=True)[1]
        # Only consider the sizes of areas that touch both edges
        area_sizes = all_area_sizes[area_ids_on_both_edges][1:]
        all_large_areas = all_area_sizes > area_sizes.mean()
        large_area_index = area_ids_on_both_edges[
            all_large_areas[area_ids_on_both_edges]
        ]  # Cursed
        largest_area_index = large_area_index[
            int((large_area_index == large_area_index.max()).nonzero()[0][0])
        ]
        size_area_id = int(largest_area_index)

        if edges_area_id == size_area_id:
            area_id = edges_area_id
        elif not all_large_areas[edges_area_id]:
            area_id = size_area_id
        else:
            msg = f"Could not determine which area is the right one. Edges: {edges_area_id}, Size: {size_area_id}."
            print(f"Large areas: {all_large_areas}")
            if plotting_level < 2:  # noqa: PLR2004
                plt.figure(figsize=(8, 6))
                plt.imshow(labeled_area, aspect="auto", origin="lower")
                plt.colorbar()
                plt.title("Areas identified")
            raise Warning(msg)
    else:
        area_id = int(area_on_both_edges.nonzero()[0][1])

    area_x_indxs, area_y_indxs = (area_id == labeled_area.T).nonzero()
    area_xy_indxs = np.vstack((area_x_indxs, area_y_indxs))

    # Find the coordinates of the boundary of the area
    one_sided_diff_edges = np.where(np.diff(area_x_indxs))[0]
    two_sided_diff_edges = np.empty(2 * len(one_sided_diff_edges) + 1, dtype=int)
    two_sided_diff_edges[0] = 0
    for i in range(1, len(two_sided_diff_edges[1:]), 2):
        two_sided_diff_edges[i] = one_sided_diff_edges[i // 2]
        two_sided_diff_edges[i + 1] = one_sided_diff_edges[i // 2] + 1
    area_edges = area_xy_indxs[:, two_sided_diff_edges]

    simple_u_grid1 = u_grid1[0, :]
    simple_u_grid2 = u_grid2[:, 0]

    area_edge_coordinates = np.vstack(
        (
            simple_u_grid1[area_edges[0, :]],
            simple_u_grid2[area_edges[1, :]] + area_shift,
        )
    )
    # Reorder coords
    linear_edge_coords = np.hstack(
        (area_edge_coordinates[:, 1::2], np.flip(area_edge_coordinates[:, ::2], 1))
    )
    area_as_path = mpl.path.Path(linear_edge_coords.T)

    if plotting_level > 1:
        # Visualize the area
        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Area")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Lines overlapping area")
        for vert in vertices:
            if any(area_as_path.contains_points(vert)):
                ax.plot(vert[:, 0], vert[:, 1], lw=0.75, marker=".", ms=1)

        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Overlapping lines. Points in area")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for vert in vertices:
            if any(area_as_path.contains_points(vert)):
                ax.plot(
                    vert[area_as_path.contains_points(vert), 0],
                    vert[area_as_path.contains_points(vert), 1],
                    lw=1,
                    marker=".",
                    ms=1.5,
                )

    # Filter contour lines that have enough points in the area to matter
    min_points_in_area_internal = 10
    relevant_line_indecies = []
    if plotting_level > 1:
        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Shortest overlapping lines removed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for i, vert in enumerate(vertices):
        if area_as_path.contains_points(vert).sum() > min_points_in_area_internal:
            relevant_line_indecies.append(i)
            if plotting_level > 1:
                ax.plot(
                    vert[area_as_path.contains_points(vert), 0],
                    vert[area_as_path.contains_points(vert), 1],
                    lw=1,
                    marker=".",
                    ms=1.5,
                )

    # Split lines with jumps into separate lines
    area_as_pathpatch = mpl.patches.PathPatch(
        area_as_path,
        edgecolor="blue",
        facecolor="None",
    )

    cropped_lines = [
        vertices[i][area_as_path.contains_points(vertices[i])]
        for i in relevant_line_indecies
    ]

    abs_diff_norm = [
        np.linalg.norm(np.diff(line, axis=0), axis=1) for line in cropped_lines
    ]
    split_vertices = []
    for i, line in enumerate(cropped_lines):
        jump_indices = (
            abs_diff_norm[i]
            > np.median(abs_diff_norm[i]) * median_separation_multiplier
        ).nonzero()[0]
        start_index = 0
        for jump_index in jump_indices:
            split_vertices.append(line[start_index:jump_index])
            start_index = jump_index + 1
        split_vertices.append(line[start_index:])

    # Filter contour lines that have enough points in the area to matter, again, and their points
    relevant_line_segments = []
    if plotting_level > 1:
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Lines with jumps split")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for vert in split_vertices:
        if area_as_path.contains_points(vert).sum() > min_points_in_area:
            relevant_line_segments.append(vert[area_as_path.contains_points(vert)])
            if plotting_level > 1:
                ax.plot(
                    vert[area_as_path.contains_points(vert), 0],
                    vert[area_as_path.contains_points(vert), 1],
                    lw=1,
                    marker=".",
                    ms=1.5,
                )

    # Filter out lines that only go in an irrelevant direction
    final_line_segments = []
    direction_diff = [
        np.diff(segment[:, 0]) for segment in relevant_line_segments
    ]
    if plotting_level > 1:
        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Only curves")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for i, diff in enumerate(direction_diff):
        if not all(diff > -0.01) and not all(diff < 0.01):  # noqa: PLR2004
            final_line_segments.append(relevant_line_segments[i])
            if plotting_level > 1:
                ax.plot(
                    relevant_line_segments[i][:, 0],
                    relevant_line_segments[i][:, 1],
                    lw=1,
                    marker=".",
                    ms=1.5,
                )

    # Select "middle" points for the lines
    selected_points = []
    for line in final_line_segments:
        positive_opening_direction = line_opens_positive(line)
        if np.isnan(positive_opening_direction):
            continue
        if positive_opening_direction:
            selected_points.append(
                line[
                    (line[:, 0] == line[:, 0].min()).nonzero()[0]
                ][0]
            )
        else:
            selected_points.append(
                line[
                    (line[:, 0] == line[:, 0].max()).nonzero()[0]
                ][0]
            )

    if plotting_level > 1:
        area_as_pathpatch = mpl.patches.PathPatch(
            area_as_path,
            edgecolor="blue",
            facecolor="None",
        )
        fig, ax = plt.subplots()
        ax.add_patch(area_as_pathpatch)
        ax.set_title("Automatically selected points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for line in final_line_segments:
            ax.plot(
                line[:, 0],
                line[:, 1],
                lw=1,
                marker=".",
                ms=1.5,
            )
        ax.plot(
            [x[0] for x in selected_points],
            [y[1] for y in selected_points],
            linestyle="None",
            marker="*",
            c="r",
        )

    if plotting_level:
        plt.figure(figsize=(8, 6))
        plt.plot(
            [x[0] for x in selected_points],
            [y[1] for y in selected_points],
            linestyle="None",
            marker="*",
            c="r",
        )
        plt.contour(u_grid1, u_grid2, kde_grid, 50)
        plt.title("Automatically selected points")
        plt.savefig("auto_points.pdf")

    if manual_point:
        if isinstance(manual_point[0], float):
            split_vertices = np.vstack([*selected_points, np.array(manual_point)])
        elif isinstance(manual_point[0], tuple | list):
            split_vertices = np.vstack(
                selected_points + [np.array(point) for point in manual_point]
            )
    else:
        split_vertices = np.vstack(selected_points)
    # Fit a polynomial to copula split
    fit_vert_2 = np.polyfit(split_vertices[:, 0], split_vertices[:, 1], deg=polydeg)
    fit_val_2 = np.poly1d(fit_vert_2)
    print("Fitted polynomial:")
    print(fit_val_2)

    # Plot the alpha sequence split in copula space
    if plotting_level:
        for j in range(2):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
            im = ax.imshow(kde_grid, aspect="auto", origin="lower", extent=(0, 1, 0, 1))
            ax.set_xlabel(r"$u_1$")
            ax.set_ylabel(r"$u_2$")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels([0, "", 0.5, "", 1])
            ax.set_yticklabels([0, "", 0.5, "", 1])
            ax.set_ylim(top=1)

            if j:
                cop_contour = ax.contour(
                    u_grid1,
                    u_grid2,
                    kde_grid,
                    levels=n_levels,
                    colors="white",
                    alpha=0.75,
                    linewidths=1,
                )
            xp = np.linspace(0, 1, 500)
            ax.scatter(
                split_vertices[:, 0],
                split_vertices[:, 1],
                color="white",
                s=75,
                zorder=4,
            )
            if manual_point:
                if isinstance(manual_point[0], float):
                    ax.scatter(
                        manual_point[0], manual_point[1], color="red", s=85, zorder=5
                    )
                else:
                    ax.scatter(
                        [point[0] for point in manual_point],
                        [point[1] for point in manual_point],
                        color="red",
                        s=85,
                        zorder=5,
                        label="manual point",
                    )
            ax.plot(
                xp, fit_val_2(xp), "--", color="white", lw=3, label="Copula Space Split"
            )
            ax.legend(fontsize=14)
            plt.colorbar(im, ax=ax, orientation="vertical", label="Copula Density")
            plt.savefig(f"copula_fit_{j}.pdf")

    # Obtain alpha sequence split in data space of original sample (using copula split)
    print("Split sample with copula")

    sequence = np.zeros(shape=(len(select_stars), 2))
    sequence[:, 0] = select_stars[xcol]
    sequence[:, 1] = select_stars[ycol]

    # Generate the inverse ECDF lambda function
    iecdf = []
    interp = []
    for i in range(2):
        sequence_sorted = np.sort(sequence[:, i])
        sequence_u_sorted = ecdf[i](sequence_sorted)
        interp.append(interp1d(sequence_u_sorted, sequence_sorted))
        iecdf.append(lambda u, i=i: interp[i](u))

    # Use the inverse ECDF to obtain data space split
    xp = np.linspace(0.001, 0.999, 500)
    dat_fit_x = iecdf[0](xp)
    if fit_val_2(xp).max() < 1.0 and fit_val_2(xp).min() > 0.0:
        dat_fit_y = iecdf[1](fit_val_2(xp))
    elif fit_val_2(xp).max() >= 1.0 and fit_val_2(xp).min() <= 0.0:
        fit_val_indxs = ((fit_val_2(xp) <= 1) & (fit_val_2(xp) >= 0)).nonzero()
        dat_fit_x = dat_fit_x[fit_val_indxs]
        dat_fit_y = iecdf[1](fit_val_2(xp)[fit_val_indxs])
    elif fit_val_2(xp).max() >= 1.0:
        fit_val_indxs = (fit_val_2(xp) <= 1).nonzero()
        dat_fit_x = dat_fit_x[fit_val_indxs]
        dat_fit_y = iecdf[1](fit_val_2(xp)[fit_val_indxs])
    elif fit_val_2(xp).min() <= 0.0:
        fit_val_indxs = (fit_val_2(xp) >= 0).nonzero()
        dat_fit_x = dat_fit_x[fit_val_indxs]
        dat_fit_y = iecdf[1](fit_val_2(xp)[fit_val_indxs])

    # Fit a polynomial to data split
    dat_fit = np.polyfit(dat_fit_x, dat_fit_y, deg=polydeg)
    dat_fit_line = np.poly1d(dat_fit)

    xp = np.linspace(-1.5, 0.75, 50)
    coords_list = [(x, dat_fit_line(x)) for x in xp]
    above_inds = line_selection_over(coords_list, select_stars, xcol=xcol, ycol=ycol)
    below_inds = line_selection_under(
        coords_list, select_stars, xcol=xcol, ycol=ycol
    )

    if plotting_level:
        plt.figure(figsize=(9, 6))
        hist = plt.hist2d(
            select_stars[xcol].value,
            select_stars[ycol].value,
            bins=300,
            norm=mpl.colors.LogNorm(),
            cmap="viridis",
            rasterized=True,
        )[3]
        plt.plot(xp, dat_fit_line(xp), linewidth=2, color="red")
        plt.colorbar(hist)
        plt.xlim(
            np.nanpercentile(select_stars[xcol].value, 0.1),
            np.nanpercentile(select_stars[xcol].value, 99.9),
        )
        plt.ylim(
            np.nanpercentile(select_stars[ycol].value, 0.1),
            np.nanpercentile(select_stars[ycol].value, 99.9),
        )
        plt.xlabel(
            f"[{xcol.split("_")[0].capitalize()}/{xcol.split("_")[1].capitalize()}]"
        )
        plt.ylabel(
            f"[{ycol.split("_")[0].capitalize()}/{ycol.split("_")[1].capitalize()}]"
        )
        plt.savefig("copula_split.pdf")
    return above_inds, below_inds, dat_fit_line
