from typing import Callable
import numpy as np
from numpy.typing import NDArray


def build_time_step_nls(stiffness: NDArray, mass: NDArray, dt: float = 00.1, param_p: int = 3, param_lambda: float = 1.0) -> Callable:

    def time_step_phi(u: NDArray, phi: NDArray) -> NDArray:
        return 2*np.abs(u)**(param_p-1) - phi

    def time_step_u(u_last: NDArray, phi: np.ndarray) -> NDArray:
        u_temp = (2j*mass - stiffness*dt - param_lambda*dt *
                  mass@np.diag(phi)) @ u_last

        return np.linalg.solve(2j*mass + stiffness*dt - param_lambda*dt*mass@np.diag(phi), u_temp)

    def time_step(u: NDArray, phi: NDArray):
        phi_new = time_step_phi(u, phi)
        u_new = time_step_u(u, phi)

        return u_new, phi_new

    return time_step
