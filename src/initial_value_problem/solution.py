import numpy as np
from numpy.typing import NDArray


def construct_solution(eigvec: NDArray, mask: NDArray) -> NDArray:
    eigenfunc = np.zeros_like(mask).astype(complex)

    i = 0
    for k in range(len(mask)):
        if not mask[k]:
            eigenfunc[k] = eigvec[i]
            i += 1

    return eigenfunc
