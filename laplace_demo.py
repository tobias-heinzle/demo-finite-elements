import os

import scipy
from scipy.special import jn_zeros
import numpy as np

import matplotlib.pyplot as plt

from src.eigen_problem.eigenfunc import construct_eigenfunc
from src.galerkin.assembly_global import assemble_matrices
from src.galerkin.util import get_nonzero_block
from src.mesh.shewchuk import construct_mesh
from src.plot.domain import plot_boundary, plot_and_save_mesh, plot_mesh
from src.plot.solution import plot_solution, generate_solution_image

np.set_printoptions(precision=2)

PLOT_PATH = "plots/laplace"

N_VALUES = 16
N_BOUNDARY = 40
TRI_AREA = round(1./250. , 5)

MODE = "circle"

print("MODE:", MODE)

mesh, region = construct_mesh(mode=MODE, tri_area=TRI_AREA, n_boundary=N_BOUNDARY)

print("vertices:", len(mesh["vertices"]))


A, M = assemble_matrices(mesh)
A, mask = get_nonzero_block(A)
M, mask_M    = get_nonzero_block(M)

assert all(
    mask == mask_M), "Nonzero block of stiffness and mass matrix do not match!"

print("A.shape:", A.shape, "\nM.shape:", M.shape)

assert N_VALUES <= A.shape[0], "# Eigenvalues of A < N_VALUES, choose finer mesh!"

eigvals, eigvecs = scipy.linalg.eigh(A, M)
eigvecs = eigvecs.T

print(f"First {N_VALUES} eigenvalues:", np.sort(sorted(eigvals)[:N_VALUES]))

plot_dir = PLOT_PATH + "/" + MODE
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

A_lim = max(np.max(A), -np.min(A))
plt.gcf().set_size_inches(12, 6)
plt.subplot(1, 2, 1)
plt.imshow(A,interpolation='none',cmap='bwr', vmin=-A_lim, vmax=A_lim)
plt.colorbar()
plt.title("A")
plt.subplot(1, 2, 2)

M_lim = max(np.max(M), -np.min(M))
plt.imshow(M,interpolation='none',cmap='bwr', vmin=-M_lim, vmax=M_lim)
plt.colorbar()
plt.title("M")
plt.tight_layout()
plt.savefig(f"{plot_dir}/bwr_matrix.png", dpi=400)
plt.close()

plt.subplot(1, 2, 1)
plt.spy(A)
plt.title("A")
plt.subplot(1, 2, 2)
plt.spy(M)
plt.title("M")
plt.tight_layout()
plt.savefig(f"{plot_dir}/spy_matrix.png")
plt.close()

plot_boundary(mesh)
plot_and_save_mesh(mesh, f"{plot_dir}/domain.png")

true_eigs = None

if MODE == "circle":
    seq = np.concatenate([np.array([jn_zeros(k,l)[-1] for l in range(1,8)]).reshape((7,1)) for k in range(9)] + [np.array([jn_zeros(k,l)[-1] for l in range(1,8)]).reshape((7,1)) for k in range(1,9)])
    seq = seq.T / 0.5
    seq.sort()
    true_eigs = seq[0]**2 # unit circle
elif MODE == "rectangle":
    true_eigs = np.array([2, 5, 5, 8, 10, 10, 13, 13, 17, 17, 18])*np.pi*np.pi # unit square


if true_eigs is not None:
    plt.plot(true_eigs[:11], 'xk', label="Exact")
    plt.plot(np.sort(eigvals)[:11], 'xr', label="Appoximate")
    plt.grid()
    plt.legend()
    plt.savefig(f"{plot_dir}/eigvals.png")
    plt.close()


eigenpairs = list(zip(eigvals, eigvecs))
eigenpairs.sort(key=lambda item: item[0])

plot_square_dim = 3

labels = False

n_plots = plot_square_dim**2

assert A.shape[0] >= n_plots, f"# Eigenvalues of A < {n_plots}, choose finer mesh if you want eigenfunction plots!"

plt.gcf().set_size_inches((n_plots, n_plots))
for k in range(n_plots):
    val, vec = eigenpairs[k]
    eigenfunc = construct_eigenfunc(vec, mask)
    X,Y,Z = generate_solution_image(eigenfunc, mesh)

    plt.subplot(plot_square_dim, plot_square_dim, k + 1)
    plot_solution(X,Y,Z,r"$\lambda = $ " + str(round(val, 2)), kind="pcolormesh")
    plot_boundary(mesh, linewidth=1.5)
    if not labels:
        plt.axis("off")
plt.tight_layout()
plt.savefig(f"{plot_dir}/eigfunc.png")
plt.close()

plt.gcf().set_size_inches((n_plots, n_plots))
for k in range(n_plots):
    val, vec = eigenpairs[k]
    eigenfunc = construct_eigenfunc(vec, mask)
    X,Y,Z = generate_solution_image(eigenfunc, mesh)

    plt.subplot(plot_square_dim, plot_square_dim, k + 1)
    plot_solution(X,Y,Z,r"$\lambda = $ " + str(round(val, 2)), kind="pcolormesh")
    plot_mesh(mesh, linewidth=0.5)
    plot_boundary(mesh, linewidth=1.5)
    if not labels:
        plt.axis("off")

plt.tight_layout()
plt.savefig(f"{plot_dir}/eigfunc_mesh.png")
plt.close()

plt.gcf().set_size_inches((n_plots, n_plots))
for k in range(n_plots):
    val, vec = eigenpairs[k]
    eigenfunc = construct_eigenfunc(vec, mask)
    X,Y,Z = generate_solution_image(eigenfunc, mesh)

    plt.subplot(plot_square_dim, plot_square_dim, k + 1)
    plot_solution(X,Y,Z,r"$\lambda = $ " + str(round(val, 2)), kind="contour")
    plot_boundary(mesh, linewidth=1.5)
    if not labels:
        plt.axis("off")

plt.tight_layout()
plt.savefig(f"{plot_dir}/eigfunc_contour.png")
plt.close()






