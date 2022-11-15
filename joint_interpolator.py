from math import floor
from numpy import array, delete, float64
from numpy.typing import NDArray

class JointInterpolator:
    _window_size: int

    def __init__(self, window_size: int):
        self._window_size = window_size

    def interpolate(self, joints: NDArray[float64]) -> NDArray[float64]:
        jsi = array(joints, copy = True)
        win_builder = JointDimWindowBuilder(joints, self._window_size)

        for joint in range(joints.shape[0]):
            for t in range(joints.shape[1]):
                for dim in range(joints.shape[2]):
                    dim_win = win_builder.capture(joint, t, dim, inclusive=True)
                    dim_mean = dim_win.mean()
                    dim_std = dim_win.std()

                    dim_val = joints[joint, t, dim]
                    dim_err = dim_val - dim_mean

                    if abs(dim_err) > 2 * dim_std:
                        dim_win_excl = win_builder.capture(joint, t, dim)
                        jsi[joint, t, dim] = dim_win_excl.mean()

        return jsi

class JointDimWindowBuilder:
    _src: NDArray[float64]
    _window_size: int

    def __init__(self, src: NDArray[float64], window_size: int):
        self._src = src
        self._window_size = window_size

    def capture(self, joint: int, time: int, dim: int, inclusive: bool = False):
        w = self._window_size
        src_max_t = self._src.shape[1]

        win_min_t = max(0, time - w)
        win_max_t = min(src_max_t, time + w)

        win = self._src[joint, win_min_t : win_max_t, dim]

        if inclusive:
            return win

        mid_pt = floor(win.size / 2)
        return delete(win, mid_pt, 0)
