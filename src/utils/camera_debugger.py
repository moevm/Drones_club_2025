import pybullet as p
import numpy as np


class CameraDebugger:

    def __init__(
        self,
        camera_position_shift: tuple = (0, 0, 0),
        camera_up_vector: tuple = (0, 0, 1),
        camera_far_distant: int = 0.5,
        camera_projection_fov: float = 60.0,
        camera_projection_aspect: float = 1.0,
        life_time: int = 0.25,
    ) -> None:
        # camera parameters
        self._camera_position_shift = np.array(camera_position_shift)
        self._camera_up_vector = camera_up_vector
        self._camera_far_distant = camera_far_distant

        self._camera_projection_fov = camera_projection_fov
        self._camera_projection_aspect = camera_projection_aspect

        # parameters for debug lines to visualize frustum
        self._life_time = life_time

    def visualize(
        self,
        drone_quaternion: list,
        drone_position: list,
    ) -> None:
        # define camera target position
        rotation_matrix = np.array(p.getMatrixFromQuaternion(drone_quaternion)).reshape(
            3, 3
        )
        target_position = np.dot(
            rotation_matrix, np.array([self._camera_far_distant, 0, 0])
        ) + np.array(drone_position)

        # get view matrix (Pybullet return them as list with 16 floats)
        view_matrix = np.array(
            p.computeViewMatrix(
                cameraEyePosition=drone_position + self._camera_position_shift,
                cameraTargetPosition=target_position,
                cameraUpVector=self._camera_up_vector,
            )
        ).reshape(4, 4)

        frustum_points = self._define_frustum_points(view_matrix)
        self._draw_frustum(frustum_points)

    def _define_frustum_points(self, view_matrix: np.array) -> list:
        # inverted matrix
        inv_view_matrix = np.linalg.inv(view_matrix)

        # define camera position
        camera_position = np.linalg.inv(view_matrix)[3, :3]

        # define view, up, and right vectors from view matrix
        view_direction = -inv_view_matrix[2, :3]
        view_direction = view_direction / np.linalg.norm(view_direction)

        up_vector = inv_view_matrix[1, :3]
        up_vector = up_vector / np.linalg.norm(up_vector)

        right_vector = np.cross(view_direction, up_vector)

        # define aspect ratio and fox from projection matrix
        aspect_ratio = self._camera_projection_aspect
        fov = np.radians(self._camera_projection_fov)  # in radians

        # define far plane points
        far_plane_half_height = self._camera_far_distant * np.tan(fov / 2)
        far_plane_half_width = far_plane_half_height * aspect_ratio
        far_plane_center = camera_position + view_direction * self._camera_far_distant
        top_left = (
            far_plane_center
            - right_vector * far_plane_half_width
            + up_vector * far_plane_half_height
        ).tolist()
        top_right = (
            far_plane_center
            + right_vector * far_plane_half_width
            + up_vector * far_plane_half_height
        ).tolist()
        bottom_left = (
            far_plane_center
            - right_vector * far_plane_half_width
            - up_vector * far_plane_half_height
        ).tolist()
        bottom_right = (
            far_plane_center
            + right_vector * far_plane_half_width
            - up_vector * far_plane_half_height
        ).tolist()

        return [camera_position, top_left, top_right, bottom_left, bottom_right]

    def _draw_frustum(self, frustum_points: list) -> None:
        (camera_position, top_left, top_right, bottom_left, bottom_right) = (
            frustum_points
        )

        # far plane lines
        far_plane_lines = [
            (top_left, top_right),
            (top_left, bottom_left),
            (bottom_left, bottom_right),
            (bottom_right, top_right),
        ]

        # lines from camera position to far plane corners
        camera_lines = [
            (camera_position, top_left),
            (camera_position, top_right),
            (camera_position, bottom_left),
            (camera_position, bottom_right),
        ]

        for points in far_plane_lines + camera_lines:
            p.addUserDebugLine(
                lineFromXYZ=points[0],
                lineToXYZ=points[1],
                lineColorRGB=[1, 0, 0],
                lifeTime=self._life_time,
            )
