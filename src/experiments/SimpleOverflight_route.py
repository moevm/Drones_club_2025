import numpy as np
from .Decision import Decision


class SimpleOverflight_route(Decision):
    def __init__(self, ctrl):
        super().__init__(ctrl)

        self.tunnel_points = [
            np.array([0.0, 0.0, 1.5]),
            np.array([25.0, 25.0, 1.5])
        ]

        self.center = np.array([30.0, 30.0])
        self.square_size = 10.0
        h = self.square_size / 2.0
        self.square_corners_xy = [
            np.array([35.0, 25.0]),
            np.array([35.0, 35.0]),
            np.array([25.0, 35.0]),
            np.array([25.0, 25.0]),
        ]

        self.base_height = 1.5
        self.height_step = 0.5
        self.max_loops = 3
        self.current_loop = 0

        self.tunnel_index = 1
        self.square_index = 0
        self.speed = 0.2
        self.tolerance = 0.8
        self.finished = False

    def _get_target_corner(self, idx):
        x, y = self.square_corners_xy[idx]
        z = self.base_height + self.current_loop * self.height_step
        return np.array([x, y, z])

    def _next_corner(self):
        self.square_index += 1
        if self.square_index >= len(self.square_corners_xy):
            self.square_index = 0
            self.current_loop += 1
            if self.current_loop >= self.max_loops:
                return False
        return True

    def update_move(self, iteration, obs, drone_serial_number, **_):
        if drone_serial_number > 0:
            pos = obs[drone_serial_number, 0:3]
            return pos, [0, 0, 0], [0, 0, 0]
        if self.finished:
            pos = obs[drone_serial_number, 0:3]
            return pos, [0, 0, 0], [0, 0, 0]

        pos = np.array(obs[drone_serial_number, 0:3])

        if self.tunnel_index < len(self.tunnel_points):
            target = self.tunnel_points[self.tunnel_index]
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            dist = np.hypot(dx, dy)

            if dist < self.tolerance:
                self.tunnel_index += 1
                yaw = self._yaw_to_center(pos)
                return pos.tolist(), [0, 0, yaw], [0, 0, 0]

            new_pos = self._move_toward(pos, target)
            yaw = np.arctan2(dy, dx) if (abs(dx) + abs(dy)) > 0 else 0
            return new_pos, [0, 0, yaw], [0, 0, 0]

        target = self._get_target_corner(self.square_index)
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < self.tolerance:
            if not self._next_corner():
                self.finished = True
                return pos.tolist(), [0, 0, self._yaw_to_center(pos)], [0, 0, 0]
            yaw = self._yaw_to_center(pos)
            return pos.tolist(), [0, 0, yaw], [0, 0, 0]

        new_pos = self._move_toward(pos, target)
        yaw = self._yaw_to_center(pos)
        return new_pos, [0, 0, yaw], [0, 0, 0]

    def _move_toward(self, pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-6:
            return pos.tolist()
        step = min(self.speed / dist, 1.0)
        new_pos = pos + np.array([dx, dy, dz]) * step
        return new_pos.tolist()

    def _yaw_to_center(self, pos):
        dx = self.center[0] - pos[0]
        dy = self.center[1] - pos[1]
        return np.arctan2(dy, dx)
