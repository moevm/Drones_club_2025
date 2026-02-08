import numpy as np


def between_points(
    NUM_WP: int, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
):
    INIT_XYZ = np.array([x0, y0, z0]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        k = (i + 1) / NUM_WP
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + (x1 - x0) * k,
            INIT_XYZ[0, 1] + (y1 - y0) * k,
            INIT_XYZ[0, 2] + (z1 - z0) * k,
        )
    return INIT_XYZ, TARGET_POS


def between_points_exp(
    NUM_WP: int, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
):
    INIT_XYZ = np.array([x0, y0, z0]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1

    for i in range(NUM_WP):
        k = 2 ** ((NUM_WP - i) / NUM_WP) - 1
        TARGET_POS[i, :] = x1 + dx * k, y1 + dy * k, z1 + dz * k
    return INIT_XYZ, TARGET_POS


def relative_movement(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float):
    return [x0 + dx, y0 + dy, z0 + dz]


def circle(
    Radius: float, NUM_WP: int, init_X: float, init_Y: float, init_Z: float, segment=1
):
    if init_Z <= 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius * np.cos(t),
            INIT_XYZ[0, 1] + Radius * np.sin(t) - Radius,
            INIT_XYZ[0, 2],
        )  # + i/NUM_WP
    return INIT_XYZ, TARGET_POS


def circle_from_center(
    Radius: float, NUM_WP: int, init_X: float, init_Y: float, init_Z: float, segment=1
):
    if init_Z <= 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y + Radius, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius * np.cos(t),
            INIT_XYZ[0, 1] + Radius * np.sin(t) - Radius,
            INIT_XYZ[0, 2],
        )  # + i/NUM_WP
    return INIT_XYZ, TARGET_POS


def circle_many_drones(
    num_drones: int,
    Radius: float,
    NUM_WP: int,
    init_X: float,
    init_Y: float,
    init_Z: float,
):
    if init_Z <= 0:
        init_Z = 0.1
    init_step = 0.1
    INIT_XYZ = np.array(
        [[init_X, init_Y, init_Z + i * init_step] for i in range(num_drones)]
    )
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius * np.cos(t),
            INIT_XYZ[0, 1] + Radius * np.sin(t) - Radius,
            INIT_XYZ[0, 2],
        )  # + i/NUM_WP
    return INIT_XYZ, TARGET_POS


def cylinder(
    Radius: float,
    NUM_WP: int,
    height: float,
    init_X: float,
    init_Y: float,
    init_Z: float,
    segment=1,
):
    if init_Z <= 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius * np.cos(t),
            INIT_XYZ[0, 1] + Radius * np.sin(t) - Radius,
            INIT_XYZ[0, 2] + height * i / NUM_WP,
        )
    return INIT_XYZ, TARGET_POS


def cylinder_frome_centre(
    Radius: float,
    NUM_WP: int,
    height: float,
    init_X: float,
    init_Y: float,
    init_Z: float,
    segment=1,
):
    if init_Z <= 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y + Radius, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius * np.cos(t),
            INIT_XYZ[0, 1] + Radius * np.sin(t) - Radius,
            INIT_XYZ[0, 2] + height * i / NUM_WP,
        )
    return INIT_XYZ, TARGET_POS


def ellipse(
    Radius_x: float,
    Radius_y: float,
    NUM_WP: int,
    init_X: float,
    init_Y: float,
    init_Z: float,
    segment=1,
):
    if init_Z == 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + Radius_x * np.cos(t),
            INIT_XYZ[0, 1] + Radius_y * np.sin(t) - Radius_y,
            INIT_XYZ[0, 2],
        )  # + i/NUM_WP
    return INIT_XYZ, TARGET_POS


def cone(
    Radius: float,
    NUM_WP: int,
    height: float,
    init_X: float,
    init_Y: float,
    init_Z: float,
    segment=1,
):
    if init_Z == 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + (1 - i / NUM_WP) * Radius * np.cos(t),
            INIT_XYZ[0, 1] + (1 - i / NUM_WP) * (Radius * np.sin(t)) - Radius,
            INIT_XYZ[0, 2] + height * i / NUM_WP,
        )
    return INIT_XYZ, TARGET_POS


def cone_frome_centre(
    Radius: float,
    NUM_WP: int,
    height: float,
    init_X: float,
    init_Y: float,
    init_Z: float,
    segment=1,
):
    if init_Z == 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y + Radius, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + (1 - i / NUM_WP) * Radius * np.cos(t),
            INIT_XYZ[0, 1] + (1 - i / NUM_WP) * (Radius * np.sin(t)) - Radius,
            INIT_XYZ[0, 2] + height * i / NUM_WP,
        )
    return INIT_XYZ, TARGET_POS


def spiral(
    Radius: float, NUM_WP: int, init_X: float, init_Y: float, init_Z: float, segment=1
):
    if init_Z == 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + (1 - i / NUM_WP) * Radius * np.cos(t),
            INIT_XYZ[0, 1] + (1 - i / NUM_WP) * (Radius * np.sin(t)) - Radius,
            INIT_XYZ[0, 2],
        )
    return INIT_XYZ, TARGET_POS


def spiral_frome_centre(
    Radius: float, NUM_WP: int, init_X: float, init_Y: float, init_Z: float, segment=1
):
    if init_Z == 0:
        init_Z = 0.1
    INIT_XYZ = np.array([init_X, init_Y + Radius, init_Z]).reshape(1, 3)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        t = (segment * 2 * np.pi * i) / NUM_WP + np.pi / 2
        TARGET_POS[i, :] = (
            INIT_XYZ[0, 0] + (1 - i / NUM_WP) * Radius * np.cos(t),
            INIT_XYZ[0, 1] + (1 - i / NUM_WP) * (Radius * np.sin(t)) - Radius,
            INIT_XYZ[0, 2],
        )
    return INIT_XYZ, TARGET_POS


def spherical_to_cartesian(R: float, phi: float, teta: float):  # angles in radians
    """
    conversion from Cartesian to spherical coordinate system
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    x = R * np.cos(teta) * np.sin(phi)
    y = R * np.sin(teta) * np.cos(phi)
    z = R * np.cos(teta)
    return [x, y, z]


def cylindrical_to_cartesian(ro: float, phi: float, z: float):  # angles in radians
    """
    conversion from Cartesian to cylindrical coordinates
    https://en.wikipedia.org/wiki/Cylindrical_coordinate_system
    """
    x = ro * np.cos(phi)
    y = ro * np.sin(phi)
    return [x, y, z]
