from cv2 import solvePnPRansac, Rodrigues
from numpy import append, array, concatenate, float64, matmul, ndarray, reshape, zeros

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

class PoseEstimator:
    __cam_mat: ndarray[(3, 3), float64]
    __rot_tra_mat: ndarray[(3, 4), float64]
    __rot_tra_mat_inv: ndarray[(3, 4), float64]

    def __init__(self, cam_intrinsics: CameraIntrinsics):
        self.__cam_mat = cam_intrinsics.camera_matrix()
        self.__rot_tra_mat = zeros((3, 4), dtype=float64)

    def estimate(
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

        return self.__rot_tra_mat

    def project(self, point: ndarray((3, 1), dtype=float64)) -> ndarray((2, 1), dtype=float64):
        point_affn = reshape(append(point, [1.0]), (4, 1))
        point_trns = matmul(matmul(self.__cam_mat, self.__rot_tra_mat), point_affn)
        
        point_norm = array([
            point_trns[0] / point_trns[2],
            point_trns[1] / point_trns[2],
        ])

        return reshape(point_norm, (2,))

    def construct_affine(rot_mat: ndarray[(3, 3), float64], tra_vec: ndarray[(3, 1)]) -> ndarray[(3, 4), float64]:
        return concatenate((rot_mat, tra_vec), axis=1, dtype=float64)

