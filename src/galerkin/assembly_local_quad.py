import numpy as np
from numpy.typing import NDArray

from ..mesh.transformation import affine_transformation
from ..mesh.util import Triangle


QUAD_WEIGHTS = [0.225,
                0.1323941527,
                0.1323941527,
                0.1323941527,
                0.1259391805,
                0.1259391805,
                0.1259391805]

third = 0.3333333333333333333333333333
a = 0.0597158717
b = 0.4701420641
c = 0.7974269853
d = 0.1012865073

SHAPE_FN = [lambda x, y: 1 - x - y,
            lambda x, _: x,
            lambda _, y: y]


QUAD_NODES = np.array([
    [third, a, b, b, c, d, d],
    [third, b, b, a, d, d, c],
]).T


def build_local_matrices_quad(vertices: Triangle) -> tuple[NDArray, NDArray]:
    B, _ = affine_transformation(*vertices)
    B_inv = np.linalg.inv(B.T)
    B_det = np.linalg.det(B)
    area = B_det/2

    A_local = np.zeros((3, 3))
    M_local = np.zeros((3, 3))

    gradients = (B_inv @ np.array([
        [-1, 1, 0],
        [-1, 0, 1],
    ])).T

    for k, weight in enumerate(QUAD_WEIGHTS):
        dv = weight * area

        x, y = QUAD_NODES[k]

        A_local += np.array([[gradients[i] @ gradients[j] * dv
                              for j in range(3)]
                             for i in range(3)])

        M_local += np.array([[SHAPE_FN[i](x, y) * SHAPE_FN[j](x, y) * dv
                              for j in range(3)]
                             for i in range(3)])

    return A_local, M_local
