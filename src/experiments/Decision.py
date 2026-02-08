"""
obs - вектор состяния дронов, для каждого дрона есть 20 значений:
0-2 - позиция
3-6 - кватернионы
7-9 - rpy
10-12 - скорости
13-15 - угловые скорости
16-19 - last_clipped_action

Чтобы задавать движение через одну из этих характеристик - нужно синхронизировать
позицию (позиция главнее). Потому, чтобы задать движение через скорость - нужно в
желаюмую позицию (target_pos) класть значение из obs[drones_number, 0:3]

"""


class Decision:
    def __init__(self, ctrl):
        self.controller = ctrl

    def reset_controller(self):
        for controller in self.controller:
            controller.reset()

    def update_move(self, iteration, obs, **_):

        if iteration % 250 == 0:
            self.reset_controller()

        tmp_pos = obs[0, 0:3]
        tmp_vel = [0, 0, 0]

        target_vel = tmp_vel.copy()
        target_pos = tmp_pos.copy()
        target_rpy = [0, 0, 0]

        if iteration % 1000 < 250:
            target_pos = tmp_pos + 0
            target_vel[1] = 0.1
        elif iteration % 1000 < 500:
            target_pos = tmp_pos + 0
            target_vel[1] = -0.1
        elif iteration % 1000 < 750:
            target_pos[0] = tmp_pos[0] + 0.05
        else:
            target_pos[0] = tmp_pos[0] - 0.05

        return target_pos, target_rpy, target_vel