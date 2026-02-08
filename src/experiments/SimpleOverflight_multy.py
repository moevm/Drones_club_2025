import numpy as np

from .Decision import Decision


class SimpleOverflight_multy(Decision):
    def __init__(self, ctrl):
        super().__init__(ctrl)
        
    def update_move(self, iteration, obs, drone_serial_number, **_):
        
        if drone_serial_number  > 0:
            target_pos = obs[drone_serial_number, 0:3] 
            target_rpy = [0, 0, 0]
            target_vel = [0, 0, 0]
            
            return target_pos, target_rpy, target_vel

            
        segment = 1
        steps_for_circle = 1500
        num_repeat = 2
        radius = 3.5
        step_h = 0.5

        if iteration > steps_for_circle * num_repeat:
            if iteration > steps_for_circle * num_repeat + 100:
                Exit = Exception
                raise Exit("Done")

            return [0, 0, 0], [0, 0, 0], [0, 0, 0]
            

        init_xyz = [0, 0, 0.5 + step_h * (iteration // steps_for_circle)] 
        t = (segment * 2 * np.pi * iteration) / steps_for_circle + np.pi / 2
        target_pos = (
            init_xyz[0] + radius * np.cos(t),
            init_xyz[1] - (radius * np.sin(t) - radius),
            init_xyz[2],
        )
        target_rpy = [0, 0, 0]
        
        P = np.array([0, 3, 0])
        D = obs[drone_serial_number, 0:3]

        # Вектор направления
        v = P - D
        roll = 0
        pitch = np.arctan2(-v[2], np.sqrt(v[0]**2 + v[1]**2))
        yaw = np.arctan2(v[1], v[0])   
        
        target_rpy = [0, 0, yaw]
        target_vel = [0, 0, 0]
        
        return target_pos, target_rpy, target_vel
    
    '''
        Запуск с gui и с фото с дрона
    python main.py --decision_name=SimpleOverflight_multy --experiment_name=simple_house --save_images=True --vision_attributes=True

        Запуск без gui и с фото с дрона
    python main.py --decision_name=SimpleOverflight_multy --experiment_name=simple_house --save_images=True --vision_attributes=True --gui=False 

        Запуск с gui и без фото
    python main.py --decision_name=SimpleOverflight_multy --experiment_name=simple_house
    '''