
import matplotlib.pyplot as plt


def plot_boundary(mesh: dict, linewidth: float = 5.0) -> None:
    for edge in mesh["segments"]:
        x = [mesh['vertices'][edge[0]][0],
             mesh['vertices'][edge[1]][0]]
        y = [mesh['vertices'][edge[0]][1],
             mesh['vertices'][edge[1]][1]]
        plt.plot(x, y, color='k', linewidth=linewidth)


def plot_mesh(mesh: dict, linewidth: float = 1.0) -> None:
    plt.triplot([mesh['vertices'][i][0] for i in range(len(mesh['vertices']))],
                [mesh['vertices'][i][1] for i in range(len(mesh['vertices']))],
                mesh['triangles'], color='k', linewidth=linewidth)


def plot_and_save_mesh(mesh: dict, path: str) -> None:
    plt.gcf().set_size_inches(6, 4)
    plot_boundary(mesh, linewidth=2)
    plot_mesh(mesh)
    plt.savefig(path, dpi=300)
    plt.close()
