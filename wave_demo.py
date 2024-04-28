import numpy as np

import matplotlib.pyplot as plt

from src.initial_value_problem.wave import build_time_step_wave
from src.initial_value_problem.projection import project_initial_condition
from src.initial_value_problem.solution import construct_solution
from src.galerkin.assembly_global import assemble_matrices
from src.galerkin.util import get_nonzero_block
from src.mesh.shewchuk import construct_mesh
from src.plot.domain import plot_boundary, plot_and_save_mesh
from src.plot.solution import plot_solution, generate_solution_image
from src.plot.gif import generate_gif

np.set_printoptions(precision=2)

MODE = "large_circle"
PLOT_PATH = "plots/wave"

N_BOUNDARY = 100
TRI_AREA = round(1./75., 5)


STEP = 0.02
T = 10.0


x_range = (-5, 5)
y_range = (-5, 5)


def u_0_func(x): return 2*np.sin(np.sqrt(x[0]**2 + x[1]**2)*2*np.pi) if (x[0]**2 + x[1]**2) <= 2 else 0
def v_0_func(x): return 0


def plot_and_save(u, t, path):

    full_solution = construct_solution(u, mask)

    plt.gcf().set_size_inches(4.75, 4)

    X, Y, Z = generate_solution_image(
        full_solution, mesh, x_range=x_range, y_range=y_range)
    plot_solution(X, Y, Z, title=r"$u_{tt} = \Delta u$ solution $u$" + f" at t = {round(t, 3)}", kind="pcolormesh", vmin=-2, vmax=2)
    plot_boundary(mesh, linewidth=1.5)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


print("MODE:", MODE)

mesh, region = construct_mesh(mode=MODE, tri_area=TRI_AREA, n_boundary=N_BOUNDARY)
plot_and_save_mesh(mesh=mesh, path=PLOT_PATH + "/domain.png")
print("vertices:", len(mesh["vertices"]))

A, M = assemble_matrices(mesh)
A, mask = get_nonzero_block(A)
M, mask_M = get_nonzero_block(M)

assert all(
    mask == mask_M), "Nonzero block of stiffness and mass matrix do not match!"

print("A.shape:", A.shape, "\nM.shape:", M.shape)

U_0 = project_initial_condition(mesh["vertices"], u_0_func)
V_0 = project_initial_condition(mesh["vertices"], v_0_func)
u = U_0[~mask]
v = V_0[~mask]
kinetic = [(v.T @ M @ v)/2 ]
potential = [(u.T @ A @ u)/2]
energy = [(v.T @ M @ v)/2 + (u.T @ A @ u)/2]


print("u.shape:", u.shape)
print("v.shape:", v.shape)
plot_and_save(u, 0, PLOT_PATH + "/frames/frame0.png")

time_step = build_time_step_wave(A, M, STEP)

for k, t in enumerate(np.arange(0, T, STEP)):
    print(f"STEP {k} (time {t}) in progress.                      ", end="\r")

    u, v = time_step(u, v)

    kinetic += [(v.T @ M @ v)/2 ]
    potential += [(u.T @ A @ u)/2]
    energy += [(v.T @ M @ v)/2 + (u.T @ A @ u)/2]
    

    plot_and_save(u, t, PLOT_PATH + f"/frames/frame{k + 1}.png")

plt.plot(np.arange(0, T + STEP, STEP), kinetic, "r", label="kinetic")
plt.plot(np.arange(0, T + STEP, STEP), potential, "b", label="potential")
plt.plot(np.arange(0, T + STEP, STEP), energy, "k", label="total")
plt.title("Energy")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH + "/energy.png")
plt.close()

print("Done")
gif = input("Generate GIF? (YES or NO)")

if gif == "YES":
    generate_gif(PLOT_PATH + "/frames", PLOT_PATH + "/animation.gif")