
import numpy as np

from ..mesh.util import get_vertices, is_on_boundary
from .assembly_local import build_local_matrices
from .assembly_local_quad import build_local_matrices_quad


def find_global_indices(triangulation, k):
    # local_indices = [(i, j) for j in range(3) for i in range(3)]
    # global_indices = []
    # for i, j in local_indices:
    #     loc2glob_i = triangulation['triangles'][k][i]
    #     loc2glob_j = triangulation['triangles'][k][j]
    #     global_indices.append((loc2glob_i, loc2glob_j))

    # return global_indices
    return [(triangulation['triangles'][k][i],
             triangulation['triangles'][k][j])
            for j in range(3) for i in range(3)]


def assemble_matrices(triangulation: dict, *, quad: bool = False):
    n_vertices = len(triangulation['vertices'])
    m_triangles = len(triangulation['triangles'])

    A_global = np.zeros((n_vertices, n_vertices))
    M_global = np.zeros((n_vertices, n_vertices))

    local_indices = [(i, j) for j in range(3) for i in range(3)]

    for k in range(m_triangles):
        vertices = get_vertices(k, triangulation)

        A_local, M_local = build_local_matrices_quad(
            vertices) if quad else build_local_matrices(vertices)

        global_indices = find_global_indices(triangulation, k)

        for j, vert in enumerate(vertices):
            # Impose Dirichlet BC
            if is_on_boundary(vert, triangulation):
                A_local[j, :] = 0.0
                A_local[:, j] = 0.0
                M_local[j, :] = 0.0
                M_local[:, j] = 0.0

        for glob, loc in zip(global_indices, local_indices):
            A_global[glob] += A_local[loc]
            M_global[glob] += M_local[loc]

    return A_global, M_global
