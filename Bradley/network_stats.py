import numpy as np
from skimage.measure import regionprops
from skimage.morphology import label
from skimage.filters import gaussian
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


def draw_line(img: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Draw a line on a numpy 2d array, from p1 to p2.

    Parameters
    ----------
    img: np.ndarray
        Numpy image to draw the line on.
    p1: np.ndarray
        First point for the line.
    p2: np.ndarray
        Second point for the line.
    """
    img = img.copy()
    x1, y1 = p1
    x2, y2 = p2
    # Swap axes if y diff is smaller than x diff
    swap = abs(x2 - x1) < abs(y2 - y1)
    if swap:
        img = img.T
        x1, y1, x2, y2 = y1, x1, y2, x2
    # Swap line direction if x2 < x1
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    # Draw line endpoints
    img[x1, y1] = True
    img[x2, y2] = True
    # Find intermediary points
    x = np.arange(x1 + 1, x2)
    y = np.round(((y2 - y1) / (x2 - x1)) * (x - x1) + y1).astype(int)
    # Write intermediary points
    img[x, y] = True

    return img if not swap else img.T


def create_near_outline_mask(image_shape: tuple, nodes: np.ndarray, gaussian_sigma: int):
    """Create masks for points on the outline, and for points near the outline. Returns two numpy boolean 2d
    arrays, one for pixels that are on the outline for the grain and one for pixel near the outline for the
    grain.

    Parameters
    ----------
    image_shape: tuple
        Shape of the image containing the grain. Used for constructing a binary mask of the correct shape.
    nodes: np.ndarray
        Numpy 2d array of node points, eg: [[1, 2], [3, 8], [9, 2]]. Used for creating the dot-to-dot
        outline.
    gaussian_sigma: int
        In order to provide a fast way of getting points near the outline, instead of calculating the
        distance for each pixel, we just blur the outline. It is difficult to get a precise set distance,
        but reduces computation significantly.

    Returns
    -------
    line_mask: np.ndarray
        Binary mask of pixels that are on the outline.
    blurred_line_mask: np.ndarray
        Binary mask of pixels that are near to the outline based on a gaussian blur of the outline.
    """

    line_mask = np.zeros(image_shape)
    for index in range(len(nodes) - 1):
        p1 = nodes[index, :]
        p1 = np.round(p1).astype(int)
        p2 = nodes[index + 1, :]
        p2 = np.round(p2).astype(int)
        line_mask = draw_line(line_mask, p1, p2)

    blurred_line_mask = gaussian(line_mask, sigma=gaussian_sigma)

    return line_mask, blurred_line_mask


def distance_to_outline(outline_mask, point):
    nonzero = np.argwhere(outline_mask == True)
    diffs = nonzero - point
    dists_squared = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    return np.min(dists_squared)


def network_density_internal(
    nodes: np.ndarray, image: np.ndarray, px_to_nm: float, stepsize_px: int, kernel_size: int, gaussian_sigma: int
):
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    density_map = np.zeros((int(np.floor(image.shape[0] / stepsize_px)), int(np.floor(image.shape[1] / stepsize_px))))
    internal_density_map = np.zeros(density_map.shape)
    near_outline_density_map = np.zeros(density_map.shape)
    densities_internal = []
    distances_internal = []
    densities_near_outline = []
    distances_near_outline = []

    print(f"density map dimensions: {internal_density_map.shape}")

    outline_mask, near_outline_mask = create_near_outline_mask(image.shape, nodes, gaussian_sigma)

    for j in range(internal_density_map.shape[0]):
        for i in range(internal_density_map.shape[1]):
            x = i * stepsize_px
            y = j * stepsize_px

            density = np.median(image[y - kernel_size : y + kernel_size, x - kernel_size : x + kernel_size])
            density_map[j, i] = density

            # volume = np.sum(image[y-kernel_size:y+kernel_size, x-kernel_size:x+kernel_size])
            # density = volume / area
            # area = stepsize_px**2 * px_to_nm**2

            in_polygon = point_in_polygon(np.array([y, x]), nodes)
            if near_outline_mask[y, x]:
                near_outline = True
            else:
                near_outline = False

            # color = 'green' if in_polygon else 'red'
            # ax.plot(y, x, marker='.', color=color)
            if in_polygon and not near_outline:
                internal_density_map[j, i] = density
                densities_internal.append(density)
                distances_internal.append(distance_to_outline(outline_mask, np.array([y, x])))
            elif near_outline:
                densities_near_outline.append(density)
                distances_near_outline.append(distance_to_outline(outline_mask, np.array([y, x])))
                near_outline_density_map[j, i] = density

    # plt.plot(nodes[:, 1], nodes[:, 0], color='black')
    # plt.show()
    return (
        density_map,
        internal_density_map,
        near_outline_density_map,
        densities_internal,
        distances_internal,
        densities_near_outline,
        distances_near_outline,
    )


def get_node_centroids(binary_img) -> tuple:
    labelled = label(binary_img)
    regions = regionprops(labelled)
    points = np.ndarray((len(regions), 2))
    for props_index, props in enumerate(regions):
        points[props_index, :] = props.centroid

    return points


def point_in_polygon(point: np.ndarray, polygon: np.ndarray):
    count = 0
    x = point[0]
    y = point[1]

    for index in range(polygon.shape[0] - 1):
        x1, y1 = polygon[index, :]
        x2, y2 = polygon[index + 1, :]

        if (y < y1) != (y < y2):
            # if x is to the left of the intersection point.
            # x - x1 < (y - y1) / m
            # intersection's x-coord is x1 plus the difference of the point and
            # p1's y coord, divided by the gradient.
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                count += 1
    if count % 2 == 0:
        return False
    else:
        return True


def network_density(nodes: np.ndarray, image: np.ndarray, px_to_nm: float):
    # Step over whole image
    stepsize_px = 10
    kernel_size = 5
    density_map = np.zeros((np.floor(image.shape[0] / stepsize_px), np.floor(image.shape[1] / stepsize_px)))
    for j in range(density_map.shape[0]):
        for i in range(density_map.shape[1]):
            # Calculate local density
            area = stepsize_px**2 * px_to_nm**2
            x = i * stepsize_px
            y = j * stepsize_px
            if point_in_polygon(np.array(x, y), nodes):
                volume = np.sum(image[y - kernel_size : y + kernel_size, x - kernel_size : x + kernel_size])
                density = volume / area
                density_map[i, j] = density

    return density_map


def polygon_perimeter(points: np.ndarray):
    points = np.append(points, points[0]).reshape(-1, 2)
    perimeter = 0
    for i in range(len(points) - 1):
        point1 = points[i, :]
        point2 = points[i + 1, :]
        perimeter += np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return perimeter


def network_area(points: np.ndarray):
    """Use the shoelace algorithm to calculate the area of an arbitrary polygon defined by a set of points.

    Parameters
    ----------
    points: np.ndarray
        2D numpy array of point coordinates defining the polygon
    Returns
    -------
    float
        Area of the polygon
    """
    points = np.append(points, points[0]).reshape(-1, 2)
    area = 0
    for index in range(len(points) - 1):
        matrix = points[index : index + 2, :].T
        area += np.linalg.det(matrix)
    return np.round(area / 2, 5)


def node_stats(labelled_image: np.ndarray, image: np.ndarray):
    """Calculate various grain statistics from a labelled image.

    Parameters
    ----------
    labelled_image: np.ndarray
        Labelled image of grains. Each label is an integer value from 1 to n where
        n is the number of grains. The labels act as masks for the grains.
    image: np.ndarray
        The original image, used to calculate stats about the grains.
    Returns
    -------
    dict
        Dictionary of grain statistics for the image"""

    region_props = regionprops(label_image=labelled_image)
    areas = np.zeros(len(region_props))
    volumes = np.zeros(len(region_props))
    max_heights = np.zeros(len(region_props))
    mean_heights = np.zeros(len(region_props))
    for props_index, props in enumerate(region_props):
        areas[props_index] = props.area
        region_points = np.where(labelled_image == props.label)
        region_values = image[region_points]
        volume = np.sum(region_values)
        volumes[props_index] = volume
        max_heights[props_index] = np.max(region_values)
        mean_heights[props_index] = np.mean(region_values)

    return {
        "node areas": areas,
        "node volumes": volumes,
        "node max_heights": max_heights,
        "node mean_heights": mean_heights,
        "number of nodes": len(region_props),
    }


def is_clockwise(p_1: tuple, p_2: tuple, p_3: tuple) -> bool:
    """Function to determine if three points make a clockwise or counter-clockwise turn.

    Parameters
    ----------
    p_1: tuple
        First point to be used to calculate turn.
    p_2: tuple
        Second point to be used to calculate turn.
    p_3: tuple
        Third point to be used to calculate turn.

    Returns
    -------
    boolean
        Indicator of whether turn is clockwise.
    """

    rotation_matrix = np.array(((p_1[0], p_1[1], 1), (p_2[0], p_2[1], 1), (p_3[0], p_3[1], 1)))
    return not np.linalg.det(rotation_matrix) > 0


def get_triangle_height(base_point_1: np.array, base_point_2: np.array, top_point: np.array) -> float:
    """Returns the height of a triangle defined by the input point vectors.
    Parameters
    ----------
    base_point_1: np.ndarray
        a base point of the triangle, eg: [5, 3].

    base_point_2: np.ndarray
        a base point of the triangle, eg: [8, 3].

    top_point: np.ndarray
        the top point of the triangle, defining the height from the line between the two base points, eg: [6,10].

    Returns
    -------
    Float
        The height of the triangle - ie the shortest distance between the top point and the line between the two base points.
    """

    # Height of triangle = A/b = ||AB X AC|| / ||AB||
    a_b = base_point_1 - base_point_2
    a_c = base_point_1 - top_point
    return np.linalg.norm(np.cross(a_b, a_c)) / np.linalg.norm(a_b)


def network_feret_diameters(edges: np.ndarray) -> float:
    """Returns the minimum and maximum feret diameters for a grain.
    These are defined as the smallest and greatest distances between
    a pair of callipers that are rotating around a 2d object, maintaining
    contact at all times.

    Parameters
    ----------
    edge_points: list
        a list of the vector positions of the pixels comprising the edge of the
        grain. Eg: [[0, 0], [1, 0], [2, 1]]
    Returns
    -------
    min_feret: float
        the minimum feret diameter of the grain
    max_feret: float
        the maximum feret diameter of the grain"""

    # Sort the vectors by x coordinate then y coordinate
    sorted_indices = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_points = edges[sorted_indices]

    # Construct upper and lower hulls for the edge points.
    upper_hull = []
    lower_hull = []
    for point in sorted_points:
        # print(point)
        while len(lower_hull) > 1 and is_clockwise(lower_hull[-2], lower_hull[-1], point):
            lower_hull.pop()
        lower_hull.append(point)
        while len(upper_hull) > 1 and not is_clockwise(upper_hull[-2], upper_hull[-1], point):
            upper_hull.pop()
        upper_hull.append(point)

    upper_hull = np.array(upper_hull)
    lower_hull = np.array(lower_hull)

    # Create list of contact vertices for calipers on the antipodal hulls
    contact_points = []
    upper_index = 0
    lower_index = len(lower_hull) - 1
    min_feret = None
    while upper_index < len(upper_hull) - 1 or lower_index > 0:
        contact_points.append([lower_hull[lower_index, :], upper_hull[upper_index, :]])
        # If we have reached the end of the upper hull, continute iterating over the lower hull
        if upper_index == len(upper_hull) - 1:
            lower_index -= 1
            small_feret = get_triangle_height(
                np.array(lower_hull[lower_index + 1, :]),
                np.array(lower_hull[lower_index, :]),
                np.array(upper_hull[upper_index, :]),
            )
            if min_feret is None or small_feret < min_feret:
                min_feret = small_feret
        # If we have reached the end of the lower hull, continue iterating over the upper hull
        elif lower_index == 0:
            upper_index += 1
            small_feret = get_triangle_height(
                np.array(upper_hull[upper_index - 1, :]),
                np.array(upper_hull[upper_index, :]),
                np.array(lower_hull[lower_index, :]),
            )
            if min_feret is None or small_feret < min_feret:
                min_feret = small_feret
        # Check if the gradient of the last point and the proposed next point in the upper hull is greater than the gradient
        # of the two corresponding points in the lower hull, if so, this means that the next point in the upper hull
        # will be encountered before the next point in the lower hull and vice versa.
        # Note that the calculation here for gradients is the simple delta upper_y / delta upper_x > delta lower_y / delta lower_x
        # however I have multiplied through the denominators such that there are no instances of division by zero. The
        # inequality still holds and provides what is needed.
        elif (upper_hull[upper_index + 1, 1] - upper_hull[upper_index, 1]) * (
            lower_hull[lower_index, 0] - lower_hull[lower_index - 1, 0]
        ) > (lower_hull[lower_index, 1] - lower_hull[lower_index - 1, 1]) * (
            upper_hull[upper_index + 1, 0] - upper_hull[upper_index, 0]
        ):
            # If the upper hull is encoutnered first, increment the iteration index for the upper hull
            # Also consider the triangle that is made as the two upper hull vertices are colinear with the caliper
            upper_index += 1
            small_feret = get_triangle_height(
                np.array(upper_hull[upper_index - 1, :]),
                np.array(upper_hull[upper_index, :]),
                np.array(lower_hull[lower_index, :]),
            )
            if min_feret is None or small_feret < min_feret:
                min_feret = small_feret
        else:
            # The next point in the lower hull will be encountered first, so increment the lower hull iteration index.
            lower_index -= 1
            small_feret = get_triangle_height(
                np.array(lower_hull[lower_index + 1, :]),
                np.array(lower_hull[lower_index, :]),
                np.array(upper_hull[upper_index, :]),
            )

            if min_feret is None or small_feret < min_feret:
                min_feret = small_feret

    contact_points = np.array(contact_points)

    # Find the minimum and maximum distance in the contact points
    max_feret = None
    for point_pair in contact_points:
        dist = np.sqrt((point_pair[0, 0] - point_pair[1, 0]) ** 2 + (point_pair[0, 1] - point_pair[1, 1]) ** 2)
        if max_feret is None or max_feret < dist:
            max_feret = dist

    return min_feret, max_feret


def _rim_curvature(xs: np.ndarray, ys: np.ndarray):
    """Calculate the curvature of a set of points. Uses the standard curvature definition of the derivative of the
    tangent vector.

    Parameters:
    ----------
    xs: np.ndarray
        One dimensional numpy array of x-coordinates of the points
    ys: np.ndarray
        One dimensional numpy array of y-coordinates of the points
    Returns:
    -------
    np.ndarray
        One-dimensional numpy array of curvatures for the spline.
    """
    extension_length = xs.shape[0]
    xs_extended = np.append(xs, xs)
    xs_extended = np.append(xs_extended, xs)
    ys_extended = np.append(ys, ys)
    ys_extended = np.append(ys_extended, ys)
    dx = np.gradient(xs_extended)
    dy = np.gradient(ys_extended)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curv = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    curv = curv[extension_length : (len(curv) - extension_length)]
    return curv


def _interpolate_between_two_points(p1: np.ndarray, p2: np.ndarray, interpolation_number: int):
    """Interpolate between two points, adding in interpolation_number number of points between them.

    Parameters
    ----------
    p1: np.ndarray
        Numpy array in the form of [x, y], defining the first point to interpolate between.
    p2: np.ndarray
        Numpy array in the form of [x, y], defining the second point to interpolate between.
    interpolation_number: int
        Number of points to add between p1 and p2.

    Returns
    -------
    np.ndarray
        2D numpy array of interpolated points starting with p1, and ending in p2.
    """

    # get x and y values of each point
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        # if the two points have the same x-coordinate,
        # generate n equally spaced y-coordinates instead
        y = np.linspace(y1, y2, interpolation_number + 1)
        x = [x1] * 5
    else:
        # generate n equally spaced points between p1 and p2
        x = np.linspace(x1, x2, interpolation_number + 1)
        y = np.linspace(y1, y2, interpolation_number + 1)

    # combine x and y values to get the interpolated points
    points = np.column_stack((x[1:-1], y[1:-1]))

    return points


def _interpolate_set_of_points(points: np.ndarray, interpolation_number: int):
    """
    Takes a set of points and interoplates between them with the number of interpolated values being set by interpolation_number.

    Parameters
    ----------
    points: np.ndarray
        2D numpy array of points to interpolate between.
    interpolation_number: int
        Number of interpolated points to add between points in the input array.

    Returns
    -------
    np.ndarray
        2D numpy array of interpolated points.
    """

    interpolated_points = np.zeros(((points.shape[0] - 1) * interpolation_number + 1, 2))
    for index in range(points.shape[0]):
        interpolated_points[index * interpolation_number, :] = points[index]
    # print(interpolated)
    for index in range(points.shape[0] - 1):
        interp = _interpolate_between_two_points(
            points[index], points[index + 1], interpolation_number=interpolation_number
        )

        interpolated_points[index * interpolation_number, :] = points[index]
        for i in range(interp.shape[0]):
            interpolated_points[index * interpolation_number + i + 1, :] = interp[i]

    return interpolated_points


def interpolate_and_get_curvature(points: np.ndarray, interpolation_number: int):
    """Calculates the curvature by firstly interpolating between each point, to increase the accuracy of the curvature
    calculation.

    Parameters
    ----------
    points: np.ndarray
        2D numpy array of points.
    interpolation_number: int
        Number of points to interpolate between each data point
    Returns
    -------
    interpolated_points_curvature: np.ndarray
        1D numpy array of curvature values for the interpolated points.
    node_curvatures: np.ndarray
        1D numpy array of curvature values for just the input points, meaning that this allows direct attribution of
        curvature values to nodes. The nth curvuature value corresponds to the nth node.
    """

    # Interpolate
    interpolated_points = _interpolate_set_of_points(points, interpolation_number=interpolation_number)
    # Get curvature
    interpolated_points_curvature = _rim_curvature(interpolated_points[:, 1], interpolated_points[:, 0])
    # Obtain curvature at node
    node_curvatures = interpolated_points_curvature[::interpolation_number]

    return interpolated_points_curvature, node_curvatures, interpolated_points


def _interpolate_points_spline(points: np.ndarray, num_points: int):
    """Interpolate a set of points using a spline.

    Parameters
    ----------
    points: np.ndarray
        Nx2 Numpy array of coordinates for the points.
    num_points: int
        The number of points to return following the calculated spline.

    Returns
    -------
    interpolated_points: np.ndarray
        An Ix2 Numpy array of coordinates of the interpolated points, where I is the number of points
        specified.
    """

    x, y = splprep(points.T, u=None, s=0.0, per=1)
    x_spline = np.linspace(y.min(), y.max(), num_points)
    x_new, y_new = splev(x_spline, x, der=0)
    interpolated_points = np.array((x_new, y_new)).T
    return interpolated_points


def interpolate_spline_and_get_curvature(points: np.ndarray, interpolation_number: int):
    """Calculate the curvature for a set of points in a closed loop. Interpolates the points using a spline
    to reduce anomalies.

    Parameters
    ----------
    points: np.ndarray
        2xN Numpy array of coordinates for the points.
    interpolation_number: int
        Number of interpolation points per point. Eg: for a set of 10 points and 2 interpolation points,
        there will be 20 points in the spline.

    Returns
    -------
    interpolated_curvatures: np.ndarray
        1xN Numpy array of curvatures corresponding to the interpolated points.
    interpolated_points: np.ndarray
        2xN Numpy array of interpolated points generated from the spline of the
        original points.
    """

    # Interpolate the data using cubic splines
    num_points = interpolation_number * points.shape[0]
    interpolated_points = _interpolate_points_spline(points=points, num_points=num_points)
    x = interpolated_points[:, 0]
    y = interpolated_points[:, 1]

    # Calculate the curvature
    interpolated_curvatures = _rim_curvature(x, y)

    return interpolated_curvatures, interpolated_points


def visualise_curvature_pixel_image(curvatures: np.ndarray, points: np.ndarray, image_size: int = 100, title: str=""):
    """Visualise the curvature of a set of points using a pixel heightmap image.

    Parameters
    ----------
    curvatures: np.ndarray
        Numpy Nx1 array of curvatures for the points.
    points: np.ndarray
        Numpy Nx2 array of coordinates for the points.
    
    Returns
    -------
    None
    """

    # Construct a visualisation
    curv_img = np.zeros((image_size, image_size)) - 1
    scaling_factor = (curv_img.shape[0]*1.4) / np.max(points) / 2
    centroid = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
    for point, curvature in zip(points, curvatures):
        scaled_point = ((np.array(curv_img.shape) / 2) + (point * scaling_factor) - centroid*scaling_factor).astype(int)
        curv_img[scaled_point[0], scaled_point[1]] = curvature

    plt.imshow(np.flipud(curv_img.T), cmap="rainbow")
    plt.colorbar()
    plt.title(title)
    plt.show()

def visualise_curvature_scatter(curvatures: np.ndarray, points: np.ndarray, title: str=""):
    """Visualise the curvature of a set of points using a scatter plot with colours of the markers
    representing the curvatures of the points."""

    # Plot the points
    scatter_plot = plt.scatter(points[:, 0], points[:, 1], c=curvatures, cmap="rainbow")
    plt.title(title)
    plt.colorbar(scatter_plot)
    plt.show()

