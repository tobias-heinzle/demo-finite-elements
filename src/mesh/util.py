
import numpy as np

Vertex = tuple[float, float]
Triangle = tuple[Vertex, Vertex, Vertex]


def get_vertices(element_index: int, triangulation: dict) -> Triangle:
    element = triangulation['triangles'][element_index]

    A = triangulation['vertices'][element[0]]
    B = triangulation['vertices'][element[1]]
    C = triangulation['vertices'][element[2]]

    return A, B, C


def is_on_boundary(point: tuple, region: dict, threshold: float = 0.000001) -> bool:
    P0 = np.array(point)
    for seg in region['segments']:
        P1 = np.array(region['vertices'][seg[0]])
        P2 = np.array(region['vertices'][seg[1]])

        dist = abs((P2[0] - P1[0])*(P1[1] - P0[1]) - (P1[0] - P0[0]) *
                   (P2[1] - P1[1]))/(np.sqrt((P2[0] - P1[0])**2 + (P2[1] - P1[1])**2))

        if dist < threshold:
            if (P0 - P2) @ (P2 - P1) <= 0 and (P0 - P1) @ (P2 - P1) >= 0:
                return True

    return False


def tri_area(A: Vertex, B: Vertex, C: Vertex) -> float:
    return abs(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))/2.0
