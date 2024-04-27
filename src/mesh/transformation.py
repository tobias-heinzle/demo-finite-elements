import numpy as np
from numpy.typing import NDArray

from .util import Vertex


def affine_transformation(A: Vertex, B: Vertex, C: Vertex) -> tuple[NDArray, NDArray]:
    # Takes in the vertices A, B, C and returns the affine transformation
    # from the reference triangle to the element

    x_0, y_0 = A
    x_1, y_1 = B
    x_2, y_2 = C

    transformation = np.array(
        [[x_1 - x_0, x_2 - x_0],
         [y_1 - y_0, y_2 - y_0]])

    translation = np.array([x_0, y_0])

    return transformation, translation
