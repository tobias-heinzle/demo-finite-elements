import numpy as np

import matplotlib.pyplot as plt

from src.initial_value_problem.schroedinger import build_time_step_nls
from src.initial_value_problem.projection import project_initial_condition
from src.initial_value_problem.solution import construct_solution
from src.galerkin.assembly_global import assemble_matrices
from src.galerkin.util import get_nonzero_block
from src.mesh.shewchuk import construct_mesh
from src.plot.domain import plot_boundary, plot_and_save_mesh
from src.plot.solution import plot_solution, generate_solution_image
from src.plot.gif import generate_gif

np.set_printoptions(precision=2)

MODE = "experiment"
PLOT_PATH = "plots/schroedinger"

N_BOUNDARY = 40
TRI_AREA = round(1./70., 5)


LAMBDA = -5.0
STEP = 0.01
P = 3
T = 1.0


sigma = 1/2
x_0 = 0
y_0 = 4
K = 8

x_range = (-2, 10)
y_range = (-2, 10)


def u_0_func(x): return np.exp(-1/(2*sigma**2) *
                               ((x[0]-x_0)**2 + (x[1] - y_0)**2))*np.exp(-1.0j*K*(x[0] - x_0))


def plot_and_save(u, t, path):

    full_solution = construct_solution(u, mask)

    plt.gcf().set_size_inches(4, 12)
    plt.subplot(3, 1, 1)
    X, Y, Z = generate_solution_image(
        full_solution.real, mesh, x_range=x_range, y_range=y_range)
    plot_solution(X, Y, Z, r"$\lambda = $ " +
                  str(round(LAMBDA, 2)), kind="pcolormesh", vmin=-0.25, vmax=0.25)
    plot_boundary(mesh, linewidth=1.5)
    plt.subplot(3, 1, 2)
    X, Y, Z = generate_solution_image(
        full_solution.imag, mesh, x_range=x_range, y_range=y_range)
    plot_solution(X, Y, Z, r"$\lambda = $ " +
                  str(round(LAMBDA, 2)), kind="pcolormesh", vmin=-0.25, vmax=0.25)
    plot_boundary(mesh, linewidth=1.5)
    plt.subplot(3, 1, 3)
    X, Y, Z = generate_solution_image(
        np.abs(full_solution), mesh, x_range=x_range, y_range=y_range)
    plot_solution(X, Y, Z, r"$\lambda = $ " +
                  str(round(LAMBDA, 2)), kind="pcolormesh", vmin=-0.25, vmax=0.25)
    plot_boundary(mesh, linewidth=1.5)
    plt.suptitle(
        r"$i u_t = - \Delta u + \lambda |u|^{p-1} u$" + f" at t = {round(t, 3)}")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


print("MODE:", MODE)

mesh, region = construct_mesh(mode=MODE, tri_area=TRI_AREA)
plot_and_save_mesh(mesh=mesh, path=PLOT_PATH + "/domain.png")
print("vertices:", len(mesh["vertices"]))

A, M = assemble_matrices(mesh)
A, mask = get_nonzero_block(A)
M, mask_M = get_nonzero_block(M)

assert all(
    mask == mask_M), "Nonzero block of stiffness and mass matrix do not match!"

print("A.shape:", A.shape, "\nM.shape:", M.shape)

U_0 = project_initial_condition(mesh["vertices"], lambda x: u_0_func(x))
U_0_interior = U_0[~mask]

u = U_0_interior
phi = np.abs(u)**(P-1)

print("phi.shape:", phi.shape)
print("u.shape:", u.shape)
plot_and_save(u, 0, PLOT_PATH + "/frames/frame0.png")

time_step = build_time_step_nls(A, M, STEP, P, LAMBDA)

for k, t in enumerate(np.arange(0, T, STEP)):
    print(f"STEP {k} (time {t}) in progress.                      ", end="\r")

    u, phi = time_step(u, phi)

    plot_and_save(u, t, PLOT_PATH + f"/frames/frame{k + 1}.png")

print("Done")
gif = input("Generate GIF? (YES or NO)")

if gif == "YES":
    generate_gif(PLOT_PATH + "/frames", PLOT_PATH + "/animation.gif")