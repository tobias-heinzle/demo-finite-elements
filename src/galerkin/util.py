import numpy as np
from numpy.typing import NDArray


def get_nonzero_block(mat: NDArray) -> tuple[NDArray, NDArray]:

    mask = np.array([all(mat[k] == 0) for k in range(mat.shape[0])])
    rows = mat[~mask]
    final = rows.T[~mask]

    return np.array(final).T, mask
