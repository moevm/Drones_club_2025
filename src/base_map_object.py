import pybullet as p
from typing import Optional, Tuple


class BaseMapObject:
    """
    Base class for creating and manipulating objects in PyBullet simulation.
    Handles common functionality for physics/static objects.
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float] = (0, 0, 0),
        mass: float = 0,
        physics_client_id: int = 0,
    ):
        """
        \n Initialize base simulation object.
        \n   position - (x, y, z) initial position of object center in world coordinates.
        \n   orientation - (roll, pitch, yaw) Euler angles in radians.
        \n   mass - mass in kg (0 makes static object).
        \n   physics_client_id - physics client ID.
        """
        self._position = position
        self._orientation = orientation
        self._mass = mass
        self._physics_client_id = physics_client_id
        self._body_id: Optional[int] = None

    def create(self) -> None:
        """\n Must be implemented by derived classes to create the actual object."""
        if self._body_id is not None:
            return

        raise NotImplementedError

    def remove(self) -> None:
        """\n Remove the object from simulation if it exists."""
        if self._body_id is not None:
            p.removeBody(self._body_id, physicsClientId=self._physics_client_id)
            self._body_id = None

    def set_position(
        self,
        position: Tuple[float, float, float],
        orientation: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        \n Move object to specified position and orientation.
        \n   position - (x, y, z) new position in world coordinates.
        \n   orientation - (roll, pitch, yaw) new Euler angles in radians (optional).
        """
        if self._body_id is not None:
            orient = (
                self._euler_to_quat(orientation)
                if orientation is not None
                else self._euler_to_quat(self._orientation)
            )
            p.resetBasePositionAndOrientation(
                self._body_id, position, orient, physicsClientId=self._physics_client_id
            )
            self._position = position
            if orientation is not None:
                self._orientation = orientation

    def rotate(
        self, rotation: Tuple[float, float, float], local_coords: bool = True
    ) -> None:
        """
        \n Apply rotation to the object using Euler angles.
        \n   rotation - (roll, pitch, yaw) Euler angles in radians.
        \n   local_coords - if True, applies rotation in local coordinates.
        """
        if self._body_id is None:
            return

        if local_coords:
            # Get current orientation and combine with new rotation
            _, current_orient = p.getBasePositionAndOrientation(
                self._body_id, physicsClientId=self._physics_client_id
            )
            current_euler = p.getEulerFromQuaternion(current_orient)
            new_euler = (
                current_euler[0] + rotation[0],
                current_euler[1] + rotation[1],
                current_euler[2] + rotation[2],
            )
        else:
            new_euler = rotation

        self.set_position(self._position, new_euler)

    def _euler_to_quat(
        self, euler_angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        return p.getQuaternionFromEuler(euler_angles)
