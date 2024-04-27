import numpy as np
from numpy.typing import NDArray

from ..mesh.util import Vertex


def phi_gradients(A: Vertex, B: Vertex, C: Vertex) -> NDArray:

    gradients = np.array([[B[1] - C[1], C[1] - A[1], A[1] - B[1]],
                          [C[0] - B[0], A[0] - C[0], B[0] - A[0]]])

    return gradients.T
