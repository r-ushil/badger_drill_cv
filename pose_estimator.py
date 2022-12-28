from cv2 import solvePnPRansac, Rodrigues
from calib3d import Calib, Point2D, Point3D
from numpy import append, array, concatenate, float64, matmul, ndarray, reshape, zeros
from typing import Optional
import numpy as np

from katchet_board import KatchetBoard

class CameraIntrinsics:
    __focal_len: float
    __sensor_w: float
    __sensor_h: float
    __image_w: float
    __image_h: float

    def __init__(self, focal_len: float, sensor_w: float, sensor_h: float, image_w: float, image_h: float):
        self.__focal_len = focal_len
        self.__sensor_w = sensor_w
        self.__sensor_h = sensor_h
        self.__image_w = image_w
        self.__image_h = image_h

    def camera_matrix(self) -> ndarray[(3, 3), float64]:
        fx = self.__focal_len * self.__image_w / self.__sensor_w
        fy = self.__focal_len * self.__image_h / self.__sensor_h

        cx = self.__image_w / 2
        cy = self.__image_h / 2

        cam_mat = array([
            [fx, .0, cx],
            [.0, fy, cy],
            [.0, .0, 1.]
        ], dtype=float64)

        return cam_mat

    def image_dims(self) -> tuple[float, float]:
        return self.__image_w, self.__image_h

class NotLocalisedError(Exception): pass

def assert_localised(func):
    def func_assert_localised(self: "PoseEstimator", *args, **kwargs):
        if self.is_localised():
            return func(self, *args, **kwargs)
        else:
            raise NotLocalisedError("Call .localise_camera() to localise camera first!")
    return func_assert_localised

class PoseEstimator:
    __cam_calib: Optional[Calib]
    __cam_intrinsics: CameraIntrinsics
    __cam_mat: ndarray[(3, 3), float64]
    __rot_tra_mat: ndarray[(3, 4), float64]
    __rot_tra_mat_inv: ndarray[(3, 4), float64]

    def __init__(self, cam_intrinsics: CameraIntrinsics):
        self.__cam_intrinsics = cam_intrinsics
        self.__cam_mat = cam_intrinsics.camera_matrix()
        self.__rot_tra_mat = zeros((3, 4), dtype=float64)

        self.__cam_calib = None

    def compute_camera_localisation_from_katchet(self, katchet_face):
        katchet_board = KatchetBoard.from_vertices_2d(katchet_face)

        inliers = np.zeros((4, 3), dtype=np.float64)

        self.localise_camera(
            points_3d=katchet_board.get_vertices_3d(),
            points_2d=katchet_board.get_vertices_2d(),
            iterations=500,
            reprojection_err=2.0,
            inliners=inliers,
            confidence=0.95,
        )

    def localise_camera(
        self,
        points_3d: ndarray[(3, int), float64],
        points_2d: ndarray[(2, int), float64],
        iterations: int,
        reprojection_err: float,
        inliners: ndarray[(3, int), float64],
        confidence: float,
    ):
        dist_coeffs = zeros((4, 1), dtype=float64)
        rot_vec = zeros((3, 1), dtype=float64)
        tra_vec = zeros((3, 1), dtype=float64)

        solvePnPRansac(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=self.__cam_mat,
            distCoeffs=dist_coeffs,
            rvec=rot_vec,
            tvec=tra_vec,
            useExtrinsicGuess=False,
            iterationsCount=iterations,
            reprojectionError=reprojection_err,
            confidence=confidence,
            inliers=inliners,
        )

        rot_vec_inv = -rot_vec

        rot_mat, _ = Rodrigues(rot_vec)
        rot_mat_inv, _ = Rodrigues(rot_vec_inv)

        tra_vec_inv = -matmul(rot_mat_inv, tra_vec)

        affine = PoseEstimator.construct_affine(rot_mat, tra_vec)
        affine_inv = PoseEstimator.construct_affine(rot_mat_inv, tra_vec_inv)

        self.__rot_tra_mat = affine
        self.__rot_tra_mat_inv = affine_inv

        image_w, image_h = self.__cam_intrinsics.image_dims()
        self.__cam_calib = Calib(width=image_w, height=image_h, T=tra_vec, R=rot_mat, K=self.__cam_mat)

    def is_localised(self) -> bool:
        return self.__cam_calib is not None

    @assert_localised
    def project_3d_to_2d(self, point_3d: ndarray[(3, 1), float64]) -> ndarray[(2, 1), float64]:
        point_affn = reshape(append(point_3d, [1.0]), (4, 1))
        point_trns = matmul(matmul(self.__cam_mat, self.__rot_tra_mat), point_affn)
        
        point_norm = array([
            point_trns[0] / point_trns[2],
            point_trns[1] / point_trns[2],
        ])

        return reshape(point_norm, (2,))

    @assert_localised
    def project_2d_to_3d(
        self,
        point_2d: ndarray[(2, 1), float64],
        X: Optional[float] = None,
        Y: Optional[float] = None,
        Z: Optional[float] = None,
    ) -> ndarray[(3, 1), float64]:
        coordinate_count = 0

        if X is not None: coordinate_count += 1
        if Y is not None: coordinate_count += 1
        if Z is not None: coordinate_count += 1

        if coordinate_count != 1:
            raise Exception("Specify one of X, Y or Z only.")

        point_3d = self.__cam_calib.project_2D_to_3D(Point2D(point_2d), X=X, Y=Y, Z=Z)

        return point_3d.reshape((3, 1)).astype('float64')

    def construct_affine(rot_mat: ndarray[(3, 3), float64], tra_vec: ndarray[(3, 1)]) -> ndarray[(3, 4), float64]:
        return concatenate((rot_mat, tra_vec), axis=1, dtype=float64)

