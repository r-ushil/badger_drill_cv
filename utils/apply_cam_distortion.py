from click import Path, argument, command, option

from calib3d import Calib
from numpy import array

@command()
@argument("input", type=Path(exists=True))
@argument("output", type=Path(exists=False))
@option("fx", prompt="Focal Length X (px)")
@option("fy", prompt="Focal Length Y (px)")
@option("cx", prompt="Principal Point X (px)")
@option("cy", prompt="Principal Point Y (px)")
def apply_cam_distortion(fx, fy, cx, cy):
    K = array([
        [fx, .0, cx],
        [.0, fy, cy],
        [.0, .0, 1.]])

    pass

if __name__ == "main":
    apply_cam_distortion()