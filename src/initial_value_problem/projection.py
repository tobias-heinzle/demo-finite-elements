from typing import Callable
import numpy as np
from numpy.typing import NDArray

from ..mesh.util import Vertex


def project_initial_condition(vertices: list[Vertex], func: Callable[[Vertex], float]) -> NDArray:
    return np.array([func(vertex) for vertex in vertices])
