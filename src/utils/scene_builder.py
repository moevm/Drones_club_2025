#!/usr/bin/env python3

import pybullet as p
import os
import numpy as np

class SceneBuilder:
    """Класс для построения различных сцен в PyBullet"""
    
    def __init__(self, physics_client_id=0):
        self.client_id = physics_client_id
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.normpath(os.path.join(self.current_dir, "..", "..", "assets"))
        self.models_dir = os.path.join(self.assets_dir, "models")
    
    def build_scene(self, scene_type, models_config=None, obj_path=None):
        """
        Построение сцены по типу
        
        Аргументы:
            scene_type: 'house', 'hay', 'city'
            models_config: конфигурация объектов из objects.yaml
            obj_path: путь к OBJ файлу (для города)
        """
        if scene_type == "city":
            return self._build_city_scene(models_config, obj_path)
        elif scene_type == "hay":
            return self._build_hay_scene(models_config)
        else:
            print(f"🏠 Сцена по умолчанию (домики) уже загружена через objects.yaml")
            return True
    
    def _build_hay_scene(self, models_config=None):
        """Сцена со стогами сена и коридорами"""
        
        # Пол
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.1],
            rgbaColor=[0.5, 0.5, 0.4, 1],
            physicsClientId=self.client_id
        )
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.1],
            physicsClientId=self.client_id
        )
        p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.1],
                         physicsClientId=self.client_id)
        
        # Находим URDF стога
        hay_urdf = None
        if models_config:
            for model in models_config:
                if "hay" in model.get("kind", "").lower() or "bale" in model.get("kind", "").lower():
                    hay_urdf = os.path.join(self.models_dir, model.get("urdf", ""))
                    break
        
        if not hay_urdf or not os.path.exists(hay_urdf):
            # Пробуем найти в текущей директории
            hay_urdf = os.path.join(os.getcwd(), "HayBales.urdf")
            if not os.path.exists(hay_urdf):
                print("⚠️ Модель стога не найдена")
                return False
        
        def load_hay(x, y, z=0, scale=0.8):
            orientation = p.getQuaternionFromEuler([1.57, 0, 0])
            p.loadURDF(hay_urdf, [x, y, z], orientation,
                      useFixedBase=True, globalScaling=scale,
                      physicsClientId=self.client_id)
        
        def create_marker(x, y, z, color, radius=0.3):
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color,
                                        physicsClientId=self.client_id)
            p.createMultiBody(0, -1, visual, [x, y, z],
                            physicsClientId=self.client_id)
        
        def create_line(x1, y1, z1, x2, y2, z2, color, line_width=3):
            p.addUserDebugLine([x1, y1, z1], [x2, y2, z2], color,
                              lineWidth=line_width, physicsClientId=self.client_id)
        
        # Расставляем стоги
        # Коридор Север-Юг
        for y in range(-10, 11, 2):
            load_hay(-8, y, 0, 0.8)
            load_hay(8, y, 0, 0.8)
        
        # Коридор Запад-Восток
        for x in range(-8, 9, 2):
            load_hay(x, -8, 0, 0.8)
            load_hay(x, 8, 0, 0.8)
        
        # Диагональные
        for i in range(-6, 7, 2):
            load_hay(i, i, 0, 0.7)
            load_hay(i, -i, 0, 0.7)
        
        # Разбросанные
        scattered = [
            (-12, -5), (-12, 0), (-12, 5), (12, -5), (12, 0), (12, 5),
            (-5, -12), (0, -12), (5, -12), (-5, 12), (0, 12), (5, 12),
        ]
        for x, y in scattered:
            load_hay(x, y, 0, 0.7)
        
        # Маркеры и линии
        for y in range(-8, 9, 2):
            create_marker(-6, y, 1.2, [0, 1, 0, 0.7], 0.25)
            create_marker(6, y, 1.2, [0, 1, 0, 0.7], 0.25)
            create_line(-7, y, 0.1, -7, y+1, 0.1, [0, 1, 0, 1], 2)
            create_line(7, y, 0.1, 7, y+1, 0.1, [0, 1, 0, 1], 2)
        
        for x in range(-8, 9, 2):
            create_marker(x, -6, 1.2, [0, 0, 1, 0.7], 0.25)
            create_marker(x, 6, 1.2, [0, 0, 1, 0.7], 0.25)
            create_line(x, -7, 0.1, x+1, -7, 0.1, [0, 0, 1, 1], 2)
            create_line(x, 7, 0.1, x+1, 7, 0.1, [0, 0, 1, 1], 2)
        
        crossings = [(-6, -6), (6, -6), (-6, 6), (6, 6), (0, 0)]
        for x, y in crossings:
            create_marker(x, y, 1.5, [1, 1, 0, 0.8], 0.35)
        
        start_points = [(-10, -10), (10, -10), (-10, 10), (10, 10)]
        for x, y in start_points:
            create_marker(x, y, 0.5, [1, 0, 0, 0.9], 0.4)
        
        print("🌾 Сцена со стогами сена загружена")
        print("   🟢 Зеленый коридор: Север-Юг")
        print("   🔵 Синий коридор: Запад-Восток")
        print("   🟡 Желтые маркеры: Перекрестки")
        return True
    
    def _build_city_scene(self, models_config=None, obj_path=None):
        """Сцена с городом из OBJ файла"""
        
        if not obj_path or not os.path.exists(obj_path):
            # Пробуем найти через конфиг
            if models_config:
                for model in models_config:
                    if "city" in model.get("kind", "").lower():
                        obj_path = os.path.join(self.models_dir, model.get("urdf", ""))
                        obj_path = obj_path.replace(".urdf", ".obj")
                        break
        
        if not obj_path or not os.path.exists(obj_path):
            print("⚠️ OBJ файл города не найден")
            return False
        
        # Анализ размеров
        def analyze_obj_size(path):
            vertices = []
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
            if vertices:
                vertices = np.array(vertices)
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                size = max_coords - min_coords
                return size, min_coords, max_coords
            return None, None, None
        
        # Пол
        floor_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[200, 200, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self.client_id
        )
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[200, 200, 0.1],
            physicsClientId=self.client_id
        )
        p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.1],
                         physicsClientId=self.client_id)
        
        # Масштабирование
        size, min_coords, max_coords = analyze_obj_size(obj_path)
        target_size = 150.0
        if size is not None:
            max_dim = max(size)
            scale = target_size / max_dim
            z_offset = 0.1
        else:
            scale = 100.0
            z_offset = 0.0
        
        euler_angles = [1.57, 0, 1.57]
        orientation = p.getQuaternionFromEuler(euler_angles)
        
        visual_id = p.createVisualShape(
            p.GEOM_MESH, fileName=obj_path, meshScale=[scale, scale, scale],
            rgbaColor=[0.8, 0.8, 0.8, 1], physicsClientId=self.client_id
        )
        collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_path, meshScale=[scale, scale, scale],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.client_id
        )
        
        p.createMultiBody(0, collision_id, visual_id, [0, 0, z_offset], orientation,
                         physicsClientId=self.client_id)
        
        print(f"🏙️ Сцена с городом загружена! Масштаб: {scale:.3f}")
        return True
