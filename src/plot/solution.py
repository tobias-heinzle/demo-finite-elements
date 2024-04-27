
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

PlotRange = tuple[float, float]


def generate_solution_image(u_k: NDArray, mesh: dict, size: int = 300, x_range: PlotRange = (0, 1), y_range: PlotRange = (0, 1)):
    x = [vertex[0] for vertex in mesh['vertices']]
    y = [vertex[1] for vertex in mesh['vertices']]

    tri = Triangulation(x, y, mesh['triangles'])

    U = LinearTriInterpolator(tri, u_k)

    plot_x = np.zeros((size, size))
    plot_y = np.zeros((size, size))
    image = np.zeros((size, size))

    x_len = x_range[1] - x_range[0]
    y_len = y_range[1] - y_range[0]

    for i in range(size):
        for j in range(size):
            x = j/(size - 1)*x_len + x_range[0]
            y = i/(size - 1)*y_len + y_range[0]
            plot_x[i][j] = x
            plot_y[i][j] = y
            image[i][j] = U(x, y)

    return plot_x, plot_y, image


def plot_solution(plot_x: NDArray, plot_y: NDArray, image: NDArray, title: str, kind: str = "contour", vmax=None, vmin=None):
    if kind == "contour":
        plt.contour(plot_x, plot_y, image, levels=24)
    elif kind == "pcolormesh":
        plt.pcolormesh(plot_x, plot_y, image, vmin=vmin, vmax=vmax, cmap="viridis")

    plt.title(title)
