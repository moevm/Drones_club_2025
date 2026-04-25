import os
import numpy as np
import pybullet as p
import cv2
from PIL import Image
from datetime import datetime
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


class AutoAviary(CtrlAviary):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB_GND_DRAG_DW,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False,
        record_gui=False,
        save_images=False,
        save_video=False,
        obstacles=True,
        user_debug_gui=True,
        vision_attributes=False,
        output_folder="results",
        models: list = None,
        scene_type: str = "house",
        **kwargs,
    ):
        self.models = models if models else []
        self.scene_type = scene_type
        self._scene_built = False

        self.SAVE_IMAGES = save_images
        self.SAVE_VIDEO = save_video
        self.RECORD_GUI = record_gui
        self.VISION_ATTR = vision_attributes

        self.rgb = None
        self.dep = None
        self.seg = None

        if self.VISION_ATTR:
            img_res = np.array([320, 240])
            self.IMG_RES = img_res
            self.rgb = np.zeros((num_drones, img_res[1], img_res[0], 4))
            self.dep = np.ones((num_drones, img_res[1], img_res[0]))
            self.seg = np.zeros((num_drones, img_res[1], img_res[0]))

        super().__init__(
            drone_model,
            num_drones,
            neighbourhood_radius,
            initial_xyzs,
            initial_rpys,
            physics,
            pyb_freq,
            ctrl_freq,
            gui,
            record,
            obstacles,
            user_debug_gui,
            output_folder,
        )

        # Повторная установка на случай, если родитель перезаписал
        self.VISION_ATTR = vision_attributes
        self.SAVE_IMAGES = save_images
        self.SAVE_VIDEO = save_video
        self.RECORD_GUI = record_gui

        if self.SAVE_IMAGES or self.SAVE_VIDEO or self.RECORD_GUI:
            self.ONBOARD_IMG_PATH = os.path.join(
                self.OUTPUT_FOLDER,
                "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                "",
            )
            os.makedirs(self.ONBOARD_IMG_PATH, exist_ok=True)

        self.SAVE_IMAGE_DRONE_FLAGS = {id: True for id in self.DRONE_IDS}
        self.GENERATE_IMAGE_DRONE_FLAGS = {id: True for id in self.DRONE_IDS}

        if self.GUI and self.RECORD_GUI:
            os.makedirs(self.ONBOARD_IMG_PATH, exist_ok=True)
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join(self.ONBOARD_IMG_PATH, "gui_record.mp4"),
                physicsClientId=self.CLIENT,
            )

        if self.VISION_ATTR:
            self.IMG_FRAME_PER_SEC = 1
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)

            if self.IMG_CAPTURE_FREQ % self.PYB_STEPS_PER_CTRL != 0:
                raise ValueError(
                    f"[ERROR] In AutoAviary.__init__(), "
                    f"PyBullet and control frequencies incompatible "
                    f"with the desired video capture frame rate ({self.IMG_FRAME_PER_SEC}Hz)"
                )

            if self.SAVE_IMAGES or self.SAVE_VIDEO:
                for i in range(self.NUM_DRONES):
                    os.makedirs(
                        os.path.join(self.ONBOARD_IMG_PATH, f"drone_{i}"),
                        exist_ok=True,
                    )

            if self.SAVE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.VIDEO_WRITERS = []
                for i in range(self.NUM_DRONES):
                    output_path = os.path.join(
                        self.ONBOARD_IMG_PATH,
                        f"drone_{i}",
                        f"drone_{i}.mp4",
                    )
                    self.VIDEO_WRITERS.append(
                        cv2.VideoWriter(
                            filename=output_path,
                            fourcc=fourcc,
                            fps=self.IMG_FRAME_PER_SEC,
                            frameSize=self.IMG_RES,
                        )
                    )

    def render(self, mode='human', close=False):
        if hasattr(super(), 'render'):
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            result = super().render(mode, close)
            sys.stdout = old_stdout
            return result
        return None

    def _addObstacles(self):
        if self.scene_type not in ["city", "hay", "house"]:
            super()._addObstacles()

        if self._scene_built:
            return

        if self.scene_type == "hay":
            self._build_hay_scene()
            for body_id in range(p.getNumBodies(self.CLIENT)):
                info = p.getBodyInfo(body_id)
                if info and b'plane' in info[0].lower():
                    p.changeVisualShape(body_id, -1, rgbaColor=[0, 0, 0, 0])
                    break
        elif self.scene_type == "city":
            self._build_city_scene()
            for body_id in range(p.getNumBodies(self.CLIENT)):
                info = p.getBodyInfo(body_id)
                if info and b'plane' in info[0].lower():
                    p.changeVisualShape(body_id, -1, rgbaColor=[0, 0, 0, 0])
                    break
        else:
            self._build_default_scene()

        self._scene_built = True

    def _get_model_params(self):
        if isinstance(self.models, dict):
            return self.models
        elif isinstance(self.models, list) and len(self.models) > 0:
            return self.models[0]
        return {}

    def _build_default_scene(self):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        path_3d_models = os.path.dirname(SCRIPT_DIR) + "/assets/models/"

        models_list = self.models if isinstance(self.models, list) else [self.models]

        for obj in models_list:
            if not isinstance(obj, dict):
                continue
            try:
                urdf_path = f"{path_3d_models}{obj['urdf']}"
                p.loadURDF(
                    fileName=urdf_path,
                    basePosition=obj.get('position', [0, 0, 0]),
                    baseOrientation=p.getQuaternionFromEuler(obj.get('euler_orientation', [0, 0, 0])),
                    globalScaling=obj.get('global_scaling', 1),
                    physicsClientId=self.CLIENT
                )
            except Exception as e:
                print(f"❌ Ошибка загрузки {obj.get('urdf')}: {e}")

    def _build_hay_scene(self):
        params = self._get_model_params()
        hay_scale = params.get('global_scaling', 0.8)
        hay_position = params.get('position', [0, 0, 0])

        floor_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[100, 100, 0.1], rgbaColor=[0.5, 0.5, 0.4, 1],
            physicsClientId=self.CLIENT
        )
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[100, 100, 0.1],
            physicsClientId=self.CLIENT
        )
        p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.1],
                          physicsClientId=self.CLIENT)

        hay_urdf = None
        urdf_path = params.get('urdf', 'hay_bales/HayBales.urdf')
        possible_paths = [
            urdf_path,
            os.path.join(os.getcwd(), urdf_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "models", urdf_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "assets", "models", urdf_path),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                hay_urdf = path
                break
        if not hay_urdf:
            print("❌ СТОГ НЕ НАЙДЕН!")
            return

        def load_hay(x, y, z=0, scale=hay_scale):
            orientation = p.getQuaternionFromEuler([1.57, 0, 0])
            p.loadURDF(hay_urdf, [x + hay_position[0], y + hay_position[1], z + hay_position[2]],
                       orientation, useFixedBase=True, globalScaling=scale,
                       physicsClientId=self.CLIENT)

        def marker(x, y, z, color, radius=0.4):
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color,
                                         physicsClientId=self.CLIENT)
            p.createMultiBody(0, -1, visual, [x + hay_position[0], y + hay_position[1], z + hay_position[2]],
                              physicsClientId=self.CLIENT)

        def line(x1, y1, z1, x2, y2, z2, color, width=4):
            p.addUserDebugLine([x1 + hay_position[0], y1 + hay_position[1], z1 + hay_position[2]],
                               [x2 + hay_position[0], y2 + hay_position[1], z2 + hay_position[2]],
                               color, lineWidth=width, physicsClientId=self.CLIENT)

        for t in range(0, 26, 2):
            cx, cy = t, t
            load_hay(cx - 2, cy + 2, 0, hay_scale)
            load_hay(cx + 2, cy - 2, 0, hay_scale)

        for cx in range(25, 36, 2):
            if cx == 25:
                continue
            load_hay(cx, 23, 0, hay_scale)

        for cy in range(25, 36, 2):
            load_hay(37, cy, 0, hay_scale)

        for cx in range(35, 24, -2):
            load_hay(cx, 37, 0, hay_scale)

        for cy in range(35, 24, -2):
            load_hay(23, cy, 0, hay_scale)

        center_x, center_y = 30, 30
        load_hay(center_x, center_y, 0.0, hay_scale)
        load_hay(center_x, center_y, 0.8, hay_scale)
        load_hay(center_x, center_y, 1.6, hay_scale)
        marker(center_x, center_y, 2.2, [1, 1, 0, 1], 0.3)

        waypoints_3d = [
            (0, 0, 0.8), (25, 25, 0.8), (35, 25, 0.8),
            (35, 35, 0.8), (25, 35, 0.8), (25, 25, 0.8)
        ]
        colors = [
            [0, 1, 0, 1], [0, 1, 0, 1],
            [0, 0, 1, 1], [0, 0, 1, 1],
            [1, 0.5, 0, 1]
        ]
        for i in range(len(waypoints_3d) - 1):
            x1, y1, z1 = waypoints_3d[i]
            x2, y2, z2 = waypoints_3d[i + 1]
            line(x1, y1, z1, x2, y2, z2, colors[min(i, len(colors)-1)], 5)

        marker(0, 0, 1.0, [1, 0, 0, 1], 0.5)
        marker(25, 25, 1.0, [0, 1, 0, 1], 0.4)
        marker(35, 25, 1.0, [0, 1, 0, 1], 0.4)
        marker(35, 35, 1.0, [0, 0, 1, 1], 0.4)
        marker(25, 35, 1.0, [0, 0, 1, 1], 0.4)

        p.addUserDebugText("START", [hay_position[0], hay_position[1], 1.2],
                          [1, 0, 0], textSize=1.2, physicsClientId=self.CLIENT)
        p.addUserDebugText("25,25", [25 + hay_position[0], 25 + hay_position[1], 1.2],
                          [0, 1, 0], textSize=1.2, physicsClientId=self.CLIENT)
        p.addUserDebugText("35,25", [35 + hay_position[0], 25 + hay_position[1], 1.2],
                          [0, 1, 0], textSize=1.2, physicsClientId=self.CLIENT)
        p.addUserDebugText("35,35", [35 + hay_position[0], 35 + hay_position[1], 1.2],
                          [0, 0, 1], textSize=1.2, physicsClientId=self.CLIENT)
        p.addUserDebugText("25,35", [25 + hay_position[0], 35 + hay_position[1], 1.2],
                          [0, 0, 1], textSize=1.2, physicsClientId=self.CLIENT)

    def _build_city_scene(self):
        params = self._get_model_params()
        city_scale = params.get('global_scaling', 10.0)
        city_position = params.get('position', [0, 0, 0])
        city_orientation = params.get('euler_orientation', [1.57, 0, 0])
        city_urdf_path = params.get('urdf', 'beautiful_city/beautiful_city.urdf')

        p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0,
                                   physicsClientId=self.CLIENT)

        floor_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[500, 500, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self.CLIENT
        )
        floor_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[500, 500, 0.1],
            physicsClientId=self.CLIENT
        )
        p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.1],
                          physicsClientId=self.CLIENT)

        city_urdf = None
        possible_paths = [
            city_urdf_path,
            os.path.join(os.getcwd(), city_urdf_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "models", city_urdf_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "assets", "models", city_urdf_path),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                city_urdf = path
                break
        if not city_urdf:
            print("❌ URDF ГОРОДА НЕ НАЙДЕН!")
            return

        orientation = p.getQuaternionFromEuler(city_orientation)
        p.loadURDF(
            city_urdf,
            basePosition=city_position,
            baseOrientation=orientation,
            globalScaling=city_scale,
            useFixedBase=True,
            physicsClientId=self.CLIENT
        )

    def update_image_save_state(self, save_image_drones: list = [], unsave_image_drones: list = []):
        for id in save_image_drones:
            self.SAVE_IMAGE_DRONE_FLAGS[id] = True
            self.GENERATE_IMAGE_DRONE_FLAGS[id] = True
        for id in unsave_image_drones:
            self.SAVE_IMAGE_DRONE_FLAGS[id] = False

    def update_image_generate_state(self, generate_image_drones: list = [], ungenerate_image_drones: list = []):
        for id in generate_image_drones:
            self.GENERATE_IMAGE_DRONE_FLAGS[id] = True
        for id in ungenerate_image_drones:
            self.GENERATE_IMAGE_DRONE_FLAGS[id] = False

    def take_image(self):
        if not self.VISION_ATTR:
            return
        if self.step_counter % self.IMG_CAPTURE_FREQ != 0:
            return

        frame_num = int(self.step_counter / self.IMG_CAPTURE_FREQ)

        for i in range(self.NUM_DRONES):
            drone_id = i + 1
            if not (self.SAVE_IMAGE_DRONE_FLAGS.get(drone_id, False) and
                    self.GENERATE_IMAGE_DRONE_FLAGS.get(drone_id, False)):
                continue

            rgb, _, _ = self._getDroneImages(i, segmentation=False)
            self.rgb[i] = rgb

            save_dir = os.path.join(self.ONBOARD_IMG_PATH, f"drone_{i}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"img_{frame_num:05d}.png")
            img = Image.fromarray(rgb.astype("uint8"), "RGBA")
            img.save(save_path)

            if self.SAVE_VIDEO and hasattr(self, 'VIDEO_WRITERS') and len(self.VIDEO_WRITERS) > i:
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                self.VIDEO_WRITERS[i].write(img_bgr)

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        if self.IMG_RES is None:
            raise ValueError("IMG_RES is not set")

        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        local_look_dir = np.array([1000, 0, 200])
        target = np.dot(rot_mat, local_look_dir) + np.array(self.pos[nth_drone, :])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=self.L, farVal=1000.0
        )
        seg_flag = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK

        w, h, rgb, dep, seg = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=0,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=seg_flag,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.CLIENT,
        )

        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg
