import pybullet as p
from typing import Optional, Tuple

from .base_map_object import BaseMapObject


class BoxMapObject(BaseMapObject):
    """
    Box object in PyBullet simulation.
    Represents a rectangular prism with specified dimensions and visual properties.
    """

    def __init__(
        self,
        dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float] = (0, 0, 0),
        mass: float = 0,
        physics_client_id: int = 0,
    ):
        """
        \n Initialize box.
        \n   dimensions - (length, width, height) of the box in meters.
        \n   color - (R, G, B, A) color with alpha for the box.
        \n   position - (x, y, z) initial position of object center in world coordinates.
        \n   orientation - (roll, pitch, yaw) Euler angles in radians.
        \n   mass - mass in kg (0 makes static object).
        \n   physics_client_id - physics client ID.
        """
        super().__init__(position, orientation, mass, physics_client_id)
        self._dimensions = dimensions
        self._color = color
        self._collision_shape_id: Optional[int] = None
        self._visual_shape_id: Optional[int] = None

    def create(self) -> None:
        """\n Create the box in the simulation."""
        if self._body_id is not None:
            return

        half_extents = [dim / 2 for dim in self._dimensions]

        self._collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._physics_client_id,
        )

        self._visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=self._color,
            physicsClientId=self._physics_client_id,
        )

        self._body_id = p.createMultiBody(
            baseMass=self._mass,
            baseCollisionShapeIndex=self._collision_shape_id,
            baseVisualShapeIndex=self._visual_shape_id,
            basePosition=self._position,
            baseOrientation=self._euler_to_quat(self._orientation),
            physicsClientId=self._physics_client_id,
        )

    def remove(self) -> None:
        """\n Remove the box from simulation and clean up shapes."""
        super().remove()

        if self._collision_shape_id is not None:
            p.removeCollisionShape(self._collision_shape_id)
            self._collision_shape_id = None

        if self._visual_shape_id is not None:
            p.removeVisualShape(self._visual_shape_id)
            self._visual_shape_id = None
