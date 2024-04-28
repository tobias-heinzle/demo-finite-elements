from typing import Callable
import numpy as np
from numpy.typing import NDArray


def build_time_step_wave(stiffness: NDArray, mass: NDArray, dt: float = 0.1) -> Callable:

    n = stiffness.shape[0]

    B_1 = np.block([[ np.eye(n),     - dt*np.eye(n)/2 ],
                    [ dt*stiffness/2,            mass ]])
    
    B_2 = np.block([[ np.eye(n),        dt*np.eye(n)/2 ],
                    [ -dt*stiffness/2,            mass ]])

    def time_step(u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        y = np.concatenate([u, v])

        y_temp = B_2 @ y

        y_final = np.linalg.solve(B_1, y_temp)

        return y_final[:n], y_final[n:]



    return time_step
