import os
from datetime import datetime
import numpy as np
import pybullet as p
import cv2
from PIL import Image
import time
import random

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
        # physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False,
        record_gui=False,
        save_images=False,
        save_video=False,
        models: list = None,
        obstacles=True,
        user_debug_gui=True,
        vision_attributes=True,
        output_folder="auto_results",
    ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of
            the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        self.models = models if models else []

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

        self.SAVE_IMAGES = save_images
        self.SAVE_VIDEO = save_video
        self.RECORD_GUI = record_gui
        self.VISION_ATTR = vision_attributes
        if self.SAVE_IMAGES or self.SAVE_VIDEO or self.RECORD_GUI:
            self.ONBOARD_IMG_PATH = os.path.join(
                self.OUTPUT_FOLDER,
                "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
                "",
            )

        self.SAVE_IMAGE_DRONE_FLAGS = {id: True for id in self.DRONE_IDS}
        self.GENERATE_IMAGE_DRONE_FLAGS = {id: True for id in self.DRONE_IDS}

        if self.VISION_ATTR and (self.SAVE_IMAGES or self.SAVE_VIDEO):
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)

        if self.RECORD:
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)

        if self.GUI and self.RECORD_GUI:
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join(self.ONBOARD_IMG_PATH, "gui_record.mp4"),
                physicsClientId=self.CLIENT,
            )

        if self.VISION_ATTR:
            self.IMG_RES = np.array([640, 480])   # 480p
            # self.IMG_RES = np.array([1280, 720])  # hd
            # self.IMG_RES = np.array([2560, 1440])   # 2k
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(
                ((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4))
            )
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ % self.PYB_STEPS_PER_CTRL != 0:
                error_message = "[ERROR] In AutoAviary.__init__(), "
                "PyBullet and control frequencies incompatible "
                "with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC)
                logging.error(error_message)
                raise ValueError(error_message)
            if self.SAVE_IMAGES or self.SAVE_VIDEO:
                for i in range(self.NUM_DRONES):
                    os.makedirs(
                        os.path.dirname(
                            self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/"
                        ),
                        exist_ok=True,
                    )

            if self.SAVE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.VIDEO_WRITERS = []
                for i in range(self.NUM_DRONES):
                    output_path = os.path.join(
                        self.ONBOARD_IMG_PATH,
                        "drone_" + str(i),
                        "drone_" + str(i) + ".mp4",
                    )
                    self.VIDEO_WRITERS.append(
                        cv2.VideoWriter(
                            filename=output_path,
                            fourcc=fourcc,
                            fps=self.IMG_FRAME_PER_SEC,
                            frameSize=self.IMG_RES,
                        )
                    )

    ################################################################################
    
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
        if self.VISION_ATTR:
            for i in range(self.NUM_DRONES):
                drone_id = i+1
                
                if self.GENERATE_IMAGE_DRONE_FLAGS[drone_id]:
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)

                    if self.SAVE_IMAGES and self.SAVE_IMAGE_DRONE_FLAGS[drone_id]:
                        self._exportImage(
                            img_type=ImageType.RGB,  # ImageType.BW, ImageType.DEP, ImageType.SEG
                            img_input=self.rgb[i],
                            path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                            frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                        )

                        if self.SAVE_VIDEO:
                            image = Image.fromarray(self.rgb[i].astype("uint8"), "RGBA")
                            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            self.VIDEO_WRITERS[i].write(image)

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whether to compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image
            captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured
            from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image
            captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            logging.error(
                "in AutoAviary._getDroneImages(), remember to set self.IMG_RES"
                "to np.array([width, height])"
            )
            raise ValueError(
                "[ERROR] in AutoAviary._getDroneImages(), remember to set self.IMG_RES"
                "to np.array([width, height])"
            )
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT,
        )

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=self.L, farVal=1000.0
        )
        SEG_FLAG = (
            p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            if segmentation
            else p.ER_NO_SEGMENTATION_MASK
        )
        [w, h, rgb, dep, seg] = p.getCameraImage(
        width=self.IMG_RES[0],
        height=self.IMG_RES[1],
        shadow=0,  # TODO: сделать настройку теней -- вкл\выкл через аргументы
        viewMatrix=DRONE_CAM_VIEW,
        projectionMatrix=DRONE_CAM_PRO,
        flags=SEG_FLAG,
        renderer=p.ER_TINY_RENDERER, 
        physicsClientId=self.CLIENT,
    )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    def _addObstacles(self):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        path_3d_models = os.path.dirname(SCRIPT_DIR) + "/assets/models/"
        path_textures = os.path.dirname(SCRIPT_DIR) + "/assets/textures/"

        for object in self.models:
            try:
                object_id = p.loadURDF(
                    fileName=f"{path_3d_models}{object['urdf']}",
                    basePosition=object['position'],
                    baseOrientation=p.getQuaternionFromEuler(object['euler_orientation']),
                    globalScaling=object.get('global_scaling', 1),
                    physicsClientId=self.CLIENT
                )

                if object.get("texture"):
                    texture_id = p.loadTexture(f"{path_textures}{object['texture']}")
                    p.changeVisualShape(object_id, -1, textureUniqueId=texture_id)
            except Exception as e:
                print(f"Some problems when loading an object: {e}")
    
    def apply_wind(self, wind_force, wind_direction):
        """
        Apply wind force to all drones in the environment.

        Parameters
        ----------
        wind_force : float
            The magnitude of the wind force to apply.
        wind_direction : np.ndarray
            The direction of the wind force as a unit vector.
        """
        for drone_id in self.DRONE_IDS:
            force_vector = wind_force * wind_direction
            p.applyExternalForce(
                objectUniqueId=drone_id,
                linkIndex=-1,
                forceObj=force_vector,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT
            )

    def sync_debug_via_drone(self, target_pos, rpy, cameraDistance=3):
        yaw = rpy[2]
        pitch = rpy[1]
        p.resetDebugVisualizerCamera(cameraDistance=cameraDistance, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target_pos)
    

    def debug_target_pos(self, position, radius=0.1):
        '''
            Метод применим только для дебага -- он тяжелый из-за постоянного создания и удаления объектов
        '''

        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=[1, 0, 0, 1],  # Красный цвет
            radius=radius
        )

        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius
        )

        sphereBodyId = p.createMultiBody(
            baseMass=0, 
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        p.removeBody(sphereBodyId)

    '''
    Отключение спама -- todo сделать аргументом
    '''
    # def render(self, mode='human', close=False):
        # return 0
    