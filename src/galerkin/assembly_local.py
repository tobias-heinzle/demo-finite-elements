
import numpy as np
from numpy.typing import NDArray

from ..mesh.util import Triangle, tri_area
from .test_functions import phi_gradients


def build_local_stiffness(vertices: Triangle) -> NDArray:

    g_0, g_1, g_2 = phi_gradients(*vertices)
    area = tri_area(*vertices)

    return np.array([
        [g_0 @ g_0, g_0 @ g_1, g_0 @ g_2],
        [g_1 @ g_0, g_1 @ g_1, g_1 @ g_2],
        [g_2 @ g_0, g_2 @ g_1, g_2 @ g_2]
    ]) / (4*area)


def build_local_mass(vertices: Triangle) -> np.ndarray:
    area = tri_area(*vertices)

    return np.array([[area/6, area/12, area/12],
                    [area/12,  area/6, area/12],
                    [area/12,  area/12, area/6]])


def build_local_matrices(vertices: Triangle) -> tuple[np.ndarray, np.ndarray]:

    return build_local_stiffness(vertices), build_local_mass(vertices)
