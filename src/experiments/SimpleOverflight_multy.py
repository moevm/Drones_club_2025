import numpy as np

from .Decision import Decision


class SimpleOverflight_multy(Decision):
    def __init__(self, ctrl):
        super().__init__(ctrl)

    def update_move(self, iteration, obs, drone_serial_number, **_):

        if drone_serial_number > 0:
            # Второй дрон просто сидит на месте
            target_pos = obs[drone_serial_number, 0:3]
            target_rpy = [0, 0, 0]
            target_vel = [0, 0, 0]
            return target_pos, target_rpy, target_vel

        # Параметры облёта
        side_length = 3.0          # длина стороны квадрата
        height_offset = 0.5        # базовая высота облёта
        step_h = 0.5               # шаг по высоте при переходе на новый виток
        steps_per_side = 800       # шагов на одну сторону
        num_repeat = 2             # количество полных обходов квадрата

        # Общее количество шагов одного полного обхода (4 стороны)
        steps_per_loop = 4 * steps_per_side
        total_iterations = steps_per_loop * num_repeat

        # Если уже всё отработали
        if iteration >= total_iterations:
            if iteration >= total_iterations + 100:
                raise Exception("Done")

            # после финального круга просто оставаться на последней точке
            side = (total_iterations - 1) // steps_per_side % 4
            t = (total_iterations - 1) % steps_per_side / steps_per_side
            center = np.array([0, 3])  # центр квадрата — дом
            corners = [
                center + np.array([-side_length / 2, -side_length / 2]),  # ниже–слева
                center + np.array([ side_length / 2, -side_length / 2]),  # ниже–справа
                center + np.array([ side_length / 2,  side_length / 2]),  # выше–справа
                center + np.array([-side_length / 2,  side_length / 2]),  # выше–слева
            ]
            current_corner = corners[side]
            next_corner = corners[(side + 1) % 4]
            xy = current_corner + t * (next_corner - current_corner)
            final_height = height_offset + step_h * (total_iterations // steps_per_loop)
            target_pos = [xy[0], xy[1], final_height]
            target_rpy = [0, 0, 0]
            target_vel = [0, 0, 0]
            return target_pos, target_rpy, target_vel

        # Текущая высота (расти по циклам)
        current_height = height_offset + step_h * (iteration // steps_per_loop)

        # Определяем сторону квадрата и где на ней мы находимся
        side = (iteration // steps_per_side) % 4
        t = (iteration % steps_per_side) / steps_per_side  # 0..1 вдоль стороны

        # Центр квадрата — дом (даже при первых итерациях)
        center = np.array([0, 3])  # тут дом, не [0,0]

        # Вершины квадрата относительно дома (против часовой стрелки)
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

