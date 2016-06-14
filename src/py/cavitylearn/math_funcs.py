
from numbers import Number
import numpy as np
import scipy.interpolate


# Thanks! http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    :param deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random rotation. Small
    deflection => small perturbation.
    :param randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    :return Numpy array of size 3x3 representing a random rotation matrix.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    # Vx, Vy, Vz = \
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def random_euler_angles(deflection=1.0, randnums=None):
    """
    Creates 3 random rotation euler angles.

    :param deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    :param randnums: An iterable with 3 random numbers between 0.0 and 1.0. If None, calls np.random.uniform
    :return: A tuple of 3 random euler angles
    """
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    x, y, z = randnums

    phi = 2.0 * np.pi * x
    theta = np.arcsin(y)
    psi = deflection * np.pi * (2.0 * z - 1)

    return phi, theta, psi


def embed_grid(in_grid, out_grid_shape, rotation_matrix=None, dtype=np.float32):
    """
    Embed a rectangular grid within another rectangular grid.
    Match all gridpoints of the in_grid to points on the out_grid using nearest neighbor search.

    :param in_grid: Larger grid that will be embedded
    :param out_grid_shape: Shape of the target grid
    :param rotation_matrix: Rotation matrix that should be applied to the input grid
    :param dtype: Data type of the target grid
    :return: Output grid with the requested type and shape.
    """

    out_shapehalf = np.array(out_grid_shape) / 2.0
    in_shapehalf = np.array(in_grid.shape) / 2.0

    # create a list of points in the input grid
    in_coords = np.indices(in_grid.shape, dtype=int).reshape(3, -1).T

    # get values corresponding to the points
    in_gridpoint_values = in_grid[in_coords[:, 0], in_coords[:, 1], in_coords[:, 2]]

    # rotate grid points if requested
    if rotation_matrix is not None:
        coords_centered = np.matmul(in_coords - in_shapehalf, rotation_matrix.T)
    else:
        coords_centered = in_coords - in_shapehalf

    # create output grid, get grid points
    out_grid = np.zeros(out_grid_shape, dtype=dtype)
    out_coords = np.indices(out_grid.shape, dtype=int).reshape(3, -1).T

    out_gridvalues = scipy.interpolate.griddata(coords_centered, in_gridpoint_values, out_coords - out_shapehalf,
                                                fill_value=0.0, method="nearest")
    out_grid[out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]] = out_gridvalues

    return out_grid


def points_to_grid(points, shape, resolution, dtype=np.float32):
    """Embed a point cloud in a rectangular grid

    This creates a box (rectangular grid) where the point cloud specified by points is embedded.
    The point cloud is shifted by moving the barycenter close to (0,0,0).

    The points in the point cloud have to lie on a rectangular grid with the gridsize specified in resolution.

    All points outside the box will be truncated.

    :param points: A numpy array with 4 rows, where the first 3 rows are the point coordinates and the 4th row is the
    point value.
    :param shape: Shape of the output grid
    :param resolution: Grid point distance
    :param dtype: Data type
    :return: A rectangular grid with the requested shape and type, with the provided points
    """

    if len(shape) != 3 or \
            not all((x > 0 for x in shape)) or \
            not all((np.equal(np.mod(x, 1), 0.0) for x in shape)):
        raise TypeError('Shape must be a triplet of positive integers')

    if not isinstance(resolution, Number) or resolution <= 0:
        raise TypeError("Resolution must be a positive number")

    # calculate new center
    center = np.average(points[:, 0:3], axis=0)

    # shift center to lie on a resolution-boundary
    center = center - np.mod(center, resolution)

    # transform point coordinates into centered, scaled coordinates
    coords_centered_unit = (points[:, 0:3] - center) / resolution

    # create grid
    grid = np.zeros(shape, dtype=dtype)
    shapehalf = np.array(shape) / 2.0

    # shift points to center, and calculate indices for the grid
    grid_indices = np.array(coords_centered_unit + shapehalf, dtype=np.int)

    # keep only points within the box
    # points >= 0 and points < shape
    valid_grid_indices_idx = np.all(grid_indices >= 0, axis=1) & np.all(grid_indices < shape, axis=1)

    valid_point_values = points[valid_grid_indices_idx, -1]

    grid[
        grid_indices[valid_grid_indices_idx, 0],
        grid_indices[valid_grid_indices_idx, 1],
        grid_indices[valid_grid_indices_idx, 2]] = valid_point_values

    return grid