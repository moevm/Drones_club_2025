import numpy as np

from .Decision import Decision


class SimpleOverflight_multy(Decision):
    def __init__(self, ctrl):
        super().__init__(ctrl)

    def update_move(self, iteration, obs, drone_serial_number, **_):

        if drone_serial_number > 0:
            target_pos = obs[drone_serial_number, 0:3]
            target_rpy = [0, 0, 0]
            target_vel = [0, 0, 0]
            return target_pos, target_rpy, target_vel

        # Параметры квадрата
        side_length = 3.0          # длина стороны квадрата
        height_offset = 0.5        # базовая высота
        step_h = 0.5               # шаг по высоте при переходе на новый виток
        steps_per_side = 800       # шагов на одну сторону
        num_repeat = 2             # количество полных обходов квадрата

        # Общее количество шагов одного полного круга (4 стороны)
        steps_per_loop = 4 * steps_per_side

        total_iterations = steps_per_loop * num_repeat

        if iteration > total_iterations:
            if iteration > total_iterations + 100:
                raise Exception("Done")
            return [0, 0, 0], [0, 0, 0], [0, 0, 0]

        # Текущий уровень высоты (зависит от номера цикла)
        current_height = height_offset + step_h * (iteration // steps_per_loop)

        # Определяем, на какой стороне квадрата сейчас и нормализуем шаг в пределах стороны
        side = (iteration // steps_per_side) % 4
        t = (iteration % steps_per_side) / steps_per_side  # 0..1 вдоль стороны

        # Центр квадрата в плоскости xy
        center = np.array([0, 0])

        # Вершины квадрата (против часовой стрелки)
        corners = [
            center + np.array([-side_length / 2, -side_length / 2]),  # ниже–слева
            center + np.array([ side_length / 2, -side_length / 2]),  # ниже–справа
            center + np.array([ side_length / 2,  side_length / 2]),  # выше–справа
            center + np.array([-side_length / 2,  side_length / 2]),  # выше–слева
        ]

        # Текущая и следующая вершина
        current_corner = corners[side]
        next_corner = corners[(side + 1) % 4]

        # Интерполяция вдоль стороны
        xy = current_corner + t * (next_corner - current_corner)

        target_pos = [xy[0], xy[1], current_height]
        target_rpy = [0, 0, 0]
        target_vel = [0, 0, 0]

        return target_pos, target_rpy, target_vel

