import numpy as np
from rotation import Rotation, slerp
from hybrid_operations import *
from hybrid_math import *


class Transform:
    """
    Transform Class representing a 3D transformation with position and orientation.
    This class encapsulates both the translation and rotation components of a transformation.

    Attributes:
        t (np.ndarray): Translation vector of shape (1, 3)
        rot (Rotation): Rotation represented by a Rotation object
    """
    def __init__(self, t:Array = None, rot: Rotation = None):
        """
        Initialize a Transform instance.
        Args:
            t: (1,3), float, translation vector
            rot: Rotation, Rotation instance
        """
        if t is None:
            t = np.array([0., 0., 0.])
        if rot is None:
            rot = Rotation.from_mat3(np.eye(3))

        assert is_array(t), 'Translation must be an array type (Tensor or Numpy).'
        assert type(t) == rot.type, 'Rotation and Translation must be the same array type (Tensor or Numpy).'
        assert t.size == 3, 'Size of translation must be 3.'

        if t.shape == (3,): t = t.reshape(1, 3)
        
        self.t: Array = t
        self.rot: Rotation = rot
        
    @staticmethod
    def from_rot_vec_t(rot_vec: np.ndarray, t: np.ndarray) -> 'Transform':
        """
        Create a Transform from a rotation vector and translation vector.
        Args:
            rot_vec: (3,), float, rotation vector
            t: (3,), float, translation vector
        Returns:
            Transform: Transform instance
        """
        assert is_array(rot_vec), 'Rotation vector must be Array type (Tensor or Numpy)'
        assert is_array(t), 'Translation vector must be Array type (Tensor or Numpy)'
        assert rot_vec.shape == (3,), 'Invalid Shape. Rotation vector must be (3,)'
        assert t.shape == (3,), 'Invalid Shape. Translation vector must be (3,)'
        rot = Rotation.from_so3(rot_vec)
        return Transform(t, rot)

    @staticmethod
    def from_mat(mat4: np.ndarray) -> 'Transform':
        """
        Create a Transform from a 4x4 transformation matrix.
        Args:
            mat4: (4,4) or (3,4), float, transformation matrix
        Returns:
            Transform: Transform instance
        """
        assert is_array(mat4), 'mat4 must be array type (Tensor or Numpy)'
        assert mat4.shape == (4, 4) or mat4.shape == (3, 4), 'Invalid Shape. The shape of mat4 must be (4,4) or (3,4)'
        return Transform(mat4[:3, 3], Rotation.from_mat3(mat4[:3, :3]))
    
    @staticmethod
    def from_dict(dict: dict) -> 'Transform':
        """
        Create a Transform from a dictionary containing a transformation matrix.
        Args:
            dict: Dictionary containing 'camtoworld' key with transformation matrix
        Returns:
            Transform: Transform instance
        """
        assert 'camtoworld' in dict, "No Value Error. There is no Cam to World Transform in Dict."
        mat = np.array(dict['camtoworld'])
        return Transform.from_mat(mat)
    
    def rot_mat(self) -> np.ndarray:
        """
        Get the rotation matrix of the transform.
        Returns:
            np.ndarray: Rotation matrix of shape (3, 3)
        """
        return self.rot.mat()
    
    def mat34(self) -> np.ndarray:
        """
        Get the 3x4 transformation matrix of the transform.
        Returns:
            np.ndarray: 3x4 transformation matrix
        """
        return np.concatenate([self.rot_mat(), self.t.T], axis=1)
    
    def mat44(self) -> np.ndarray:
        """
        Get the 4x4 transformation matrix of the transform.
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        mat34 = self.mat34()
        last = np.array([0., 0., 0., 1.]).reshape(1, 4)
        if is_tensor(mat34):
            last = convert_tensor(last, mat34)
        return np.concatenate([mat34, last], axis=0)
    
    def rot_vec_t(self) -> tuple:
        """
        Get the rotation vector and translation vector of the transform.
        Returns:
            tuple: rotation vector and translation vector
        """
        return self.rot.so3(), self.t
     
    def skew_t(self) -> np.ndarray:
        """
        Get the skew-symmetric matrix of the translation vector.
        Returns:
            np.ndarray: Skew-symmetric matrix
        """
        return vec3_to_skew(self.t)
    
    def get_t_rot_mat(self) -> tuple:
        """
        Get the translation vector and rotation matrix of the transform.
        Returns:
            tuple: translation vector and rotation matrix
        """
        return self.t, self.rot_mat()
    
    def inverse(self) -> 'Transform':
        """
        Get the inverse of the current transform.
        Returns:
            Transform: Inverse transform
        """
        R_inv = self.rot.inverse()
        t_inv = -R_inv.apply_pts3d(self.t.T)
        return Transform(t_inv.T, R_inv)
    
    # @DeprecationWarning
    def get_origin_direction(self, rays: Array):
        """
        Get the origin and direction vectors from rays (local coordinates).
        Args:
            rays: (3, n), float, rays from origin
        Returns:
            origin: (n, 3), float, origin in world coordinates
            direction: (n, 3), float, direction vector in world coordinates
        """
        assert len(rays.shape) <= 2 and rays.shape[0] == 3, "Invalid Shape. Ray's shape must be (3, n) or (3)"
        if len(rays.shape) == 2:
            n_rays = rays.shape[1]
        else:
            rays = expand_dim(rays, 1)
            n_rays = 1
        origin = np.tile(convert_numpy(self.t),(n_rays, 1))
        origin = convert_array(origin, rays)
        direction = transpose2d(self.rot.apply_pts3d(rays)).reshape((-1, 3))
        return origin, direction

    def merge(self, transform: 'Transform') -> 'Transform':
        """
        Merge the current transform with another transform.
        Args:
            transform: Transform, another transform to merge with
        Returns:
            Transform: Merged transform
        """
        mat4 = self.mat44() @ transform.mat44()
        t = mat4[:3, 3]
        rot = Rotation.from_mat3(mat4[:3, :3])
        return Transform(t, rot)
    
    def apply_pts3d(self, pts3d: Array) -> Array:
        """
        Apply the transform to 3D points.
        Args:
            pts3d: (3, n), float, 3D points
        Returns:
            Array: Transformed 3D points
        """
        t = transpose2d(convert_array(self.t, pts3d))
        return self.rot.apply_pts3d(pts3d) + t

    def __mul__(self, other):
        """
        Define the multiplication operation for Transform and Pose.
        Args:
            other: Transform or Pose, the object to multiply with
        Returns:
            Transform or Pose: Result of the multiplication
        """
        if isinstance(other, Transform):
            return self.merge(other)
        else:
            raise ValueError("Multiplication only supported with Transform or Pose")
    
def interpolate_transform(t1: Transform, t2: Transform, alpha: float) -> Transform:
    """
    Interpolate between two transforms using linear interpolation (Lerp) and spherical linear interpolation (Slerp).
    Args:
        t1: Transform, start transform
        t2: Transform, end transform
        alpha: float, interpolation parameter (0 <= alpha <= 1)
    Returns:
        Transform: Interpolated transform
    """
    r = slerp(t1.rot, t2.rot, alpha)
    trans1, trans2 = t1.t, t2.t
    trans = trans1 * (1. - alpha) + trans2 * alpha
    return Transform(t=trans, rot=r) 