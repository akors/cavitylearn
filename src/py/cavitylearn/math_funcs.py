import numpy as np


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

