# This code uses the triangle library, which is a python wrapper around
# Jonathan Richard Shewchuk’s two-dimensional quality mesh generator and
# delaunay triangulator library
# https://pypi.org/project/triangle/
# https://www.cs.cmu.edu/~quake/triangle.html

from typing import Any
from triangle import triangulate
import numpy as np


def construct_mesh(mode: str, tri_area: float,  n_boundary: int = 20) -> tuple[dict, dict]:

    if mode == "circle":
        center = 0.5
        radius = 0.5
        vertices = np.stack(
            [
                center + np.cos(np.linspace(0, 2*np.pi,
                                n_boundary + 1)[:-1])*radius,
                center + np.sin(np.linspace(0, 2*np.pi,
                                n_boundary + 1)[:-1])*radius
            ], -1)
    if mode == "large_circle":
        radius = 5
        vertices = np.stack(
            [
                np.cos(np.linspace(0, 2*np.pi,
                                n_boundary + 1)[:-1])*radius,
                np.sin(np.linspace(0, 2*np.pi,
                                n_boundary + 1)[:-1])*radius
            ], -1)
    elif mode == "rectangle":
        vertices = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
    elif mode == "triangle":
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
    elif mode == "irregular":
        vertices = np.array([[0, 0], [0, 1], [0.25, 0.75], [
                            1, 1], [0.75, 0.5], [0.25, 0]])
    elif mode == "corner":
        vertices = np.array(
            [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]])
    elif mode == "experiment":
        vertices = np.array([[-2, -2], [3.9, -2], [3.9, 2.3], [4.1, 2.3], [4.1, -2], [
                            10, -2], [10, 10], [4.1, 10], [4.1, 5.7], [3.9, 5.7], [3.9, 10], [-2, 10]])
    else:
        raise ValueError(f"invalid mode {mode}")
    segments = np.array(
        [[p, p + 1] for p in range(len(vertices) - 1)] + [[len(vertices) - 1, 0]])

    if mode == "experiment":
        middle_vertices = [[3.9, 3], [4.1, 3], [4.1, 5], [3.9, 5]]
        length = len(vertices)
        vertices = np.array([*vertices, *middle_vertices])
        segments = np.array([*segments,
                             *[[length + p, length + p + 1]
                                 for p in range(len(middle_vertices) - 1)],
                             [len(vertices) - 1, length]
                             ])

    region = {
        "vertices": vertices,
        "segments": segments,
    }

    if mode == "experiment":
        region["holes"] = [[4, 4]]

    mesh = triangulate(region, opts=f'q32.5pDa{tri_area}')

    return mesh, region
