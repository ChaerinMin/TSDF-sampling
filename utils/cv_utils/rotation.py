from hybrid_operations import *
from hybrid_math import *
from constant import PI
from enum import Enum


class RotType(Enum):
    SO3 = ("SO3", "SO(3): Special orthogonal group")
    so3 = ("so3", "so(3): Lie Algebra of SO(3)")
    QUAT_XYZW = ("QUAT_XYZW", "Quaternion with 'xyzw' ordering")
    QUAT_WXYZ = ("QUAT_WXYZ", "Quaternion with 'wxyz' ordering")
    RPY = ("RPY", "Roll-Pitch-Yaw (Euler Angles)")
    NONE = ("NONE", "NoneType")

    @staticmethod
    def from_string(type_str: str)-> 'RotType':
        if type_str == 'SO3':
            return RotType.SO3
        elif type_str == 'so3':
            return RotType.so3
        elif type_str == 'QUAT_XYZW':
            return RotType.QUAT_XYZW
        elif type_str == 'QUAT_WXYZ':
            return RotType.QUAT_WXYZ
        elif type_str == 'RPY':
            return RotType.RPY
        else:
            return RotType.NONE

def is_SO3(x: Array) -> bool:
    """
    Check given rotation array's type is either SO3 or not.
    If the type is SO3, the shape of array is (3,3)
    """
    shape = x.shape
    if shape != (3,3): return False # invalid shape

def is_so3(x: Array) -> bool:
    """
    Check given rotation array's type is either so3 or not.
    If the type is SO3, the shape of array is (3,)
    """
    shape = x.shape
    if len(shape) > 2: return False # invalid shape

    return shape[-1] == 3 

def is_quat(x: Array) -> bool:
    """
    Check given rotation array's type is either quaternion or not.
    If the type is Quaternion, the shape of array is (4,)
    """
    shape = x.shape
    if len(shape) > 2: return False # invalid shape

    # [4] or [n,4]
    return shape[-1] == 4

def is_rpy(x: Array) -> bool:
    """
    Check given rotation array's type is either RPY or not.
    If the type is RPY, the shape of array is (3,)
    """
    shape = x.shape
    return len(shape) == 1 and shape[0] == 3

def so3_to_SO3(so3: Array) -> Array:
    """
        Transform so3 to Rotation Matrix(SO3)
        Args:
            so3:(3,), float, so3
        return:
            Mat:(3,3), float, Rotation Matrix
    """
    assert is_so3(so3) and len(so3.shape) == 1, (f"Invaild Shape. so3 must be (3), but got {str(so3.shape)}")
    theta = sqrt(so3[0]**2 + so3[1]**2 + so3[2]**2)
    vec = so3 / (theta + 1e-15) 
    skew_vec = vec3_to_skew(vec)
    return exponential_map(skew_vec*theta)

def quat_to_SO3(quat: Array, is_xyzw: bool) -> Array:
    """
        Transform Quaternion to Rotation Matrix(SO3)
        Args:
            quat:(4,), float, quaternion
            is_xyzw: bool, quat is real part first. Otherwise, real part is last channel
        return:
            Mat:(3,3), float, Rotation Matrix
    """
    if is_xyzw: # real part last
        x, y, z, w = quat[0],quat[1],quat[2],quat[3] 
    else:
        w, x, y, z = quat[0],quat[1],quat[2],quat[3] 

    # Compute the elements of the rotation matrix
    m00 = 1 - 2 * (y**2 + z**2)
    m01 = 2 * (x * y + z * w)
    m02 = 2 * (x * z - y * w)
    
    m10 = 2 * (x * y - z * w)
    m11 = 1 - 2 * (x**2 + z**2)
    m12 = 2 * (y * z + x * w)

    m20 = 2 * (x * z + y * w)
    m21 = 2 * (y * z - x * w)
    m22 = 1 - 2 * (x**2 + y**2)
    
    # Create the rotation matrices for each quaternion
    rotation_matrix = stack([np.array([m00, m01, m02]),
                              np.array([m10, m11, m12]),
                              np.array([m20, m21, m22])], dim=1)

    return rotation_matrix

def rpy_to_SO3(rpy: Array) -> Array:
    """
    Transform RPY (Roll, Pitch, Yaw) to Rotation Matrix (SO3)
    Args:
        rpy: (3,), float, RPY angles
    return:
        Mat: (3,3), float, Rotation Matrix
    """
    assert is_rpy(rpy), (f"Invalid Shape. RPY must be (3,), but got {str(rpy.shape)}")
    roll, pitch, yaw = rpy

    # Calculate rotation matrix components
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    m00 = cy * cp
    m01 = cy * sp * sr - sy * cr
    m02 = cy * sp * cr + sy * sr

    m10 = sy * cp
    m11 = sy * sp * sr + cy * cr
    m12 = sy * sp * cr - cy * sr

    m20 = -sp
    m21 = cp * sr
    m22 = cp * cr

    # Create the rotation matrix
    rotation_matrix = convert_array([[m00, m01, m02],
                             [m10, m11, m12],
                             [m20, m21, m22]], rpy)
    return rotation_matrix 

def SO3_to_so3(SO3: Array) -> Array:
    assert(is_SO3(SO3)), (f"Invaild Shape. SO3 must be (3,3), but got {str(SO3.shape)}")
    
    theta = arcos((trace(SO3) - 1.)*0.5)
    vec = 0.5 / sin(theta) * stack([SO3[2,1]-SO3[1,2],SO3[0,2] - SO3[2,0],SO3[1,0] - SO3[0,1]], 0)   
    return theta * vec

def SO3_to_quat(SO3: Array) -> Array:
     # Extract the elements of the rotation matrix
    r11, r12, r13 = SO3[0, 0], SO3[0, 1], SO3[0, 2]
    r21, r22, r23 = SO3[1, 0], SO3[1, 1], SO3[1, 2]
    r31, r32, r33 = SO3[2, 0], SO3[2, 1], SO3[2, 2]

    # Calculate the quaternion components
    trace = r11 + r22 + r33
    if trace > 0:
        S = sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (r32 - r23) / S
        y = (r13 - r31) / S
        z = (r21 - r12) / S
    elif (r11 > r22) and (r11 > r33):
        S = sqrt(1.0 + r11 - r22 - r33) * 2
        w = (r32 - r23) / S
        x = 0.25 * S
        y = (r12 + r21) / S
        z = (r13 + r31) / S
    elif r22 > r33:
        S = sqrt(1.0 + r22 - r11 - r33) * 2
        w = (r13 - r31) / S
        x = (r12 + r21) / S
        y = 0.25 * S
        z = (r23 + r32) / S
    else:
        S = sqrt(1.0 + r33 - r11 - r22) * 2
        w = (r21 - r12) / S
        x = (r13 + r31) / S
        y = (r23 + r32) / S
        z = 0.25 * S

    # return concat([w, x, y, z],dim=0)
    return np.array([w, x, y, z])

def SO3_to_rpy(SO3: Array) -> Array:
    """
    Transform Rotation Matrix (SO3) to RPY (Roll, Pitch, Yaw)
    Args:
        SO3: (3,3), float, Rotation Matrix
    return:
        rpy: (3,), float, RPY angles
    """
    assert is_SO3(SO3), (f"Invalid Shape. SO3 must be (3,3), but got {str(SO3.shape)}")

    # Extract rotation matrix components
    m00, m01, m02 = SO3[0, 0], SO3[0, 1], SO3[0, 2]
    m10, m11, m12 = SO3[1, 0], SO3[1, 1], SO3[1, 2]
    m20, m21, m22 = SO3[2, 0], SO3[2, 1], SO3[2, 2]

    # Calculate RPY angles
    if m20 != 1 and m20 != -1:
        pitch = -arcsin(m20)
        roll = arctan2(m21 / cos(pitch), m22 / cos(pitch))
        yaw = arctan2(m10 / cos(pitch), m00 / cos(pitch))
    else:
        yaw = 0
        if m20 == -1:
            pitch = PI / 2
            roll = yaw + arctan2(m01, m02)
        else:
            pitch = -PI / 2
            roll = -yaw + arctan2(-m01, -m02)

    return convert_array([roll, pitch, yaw], SO3)

class Rotation:
    """
    Rotation Class. This can be recieved one of types [SO3, so3, quat].
    - if SO3(Rotation matrix), shape must be [3, 3]
    - if so3(axis angle), shape must be [3]
    - if quat(Quaternion), shape must be [4]
    - if rpy (Roll-Pitch-Yaw), shape must be [3]
    - default type is SO3 and default shape is [3, 3]
    """  
    def __init__(self, data: Array, rot_type: RotType) -> None:
        
        assert is_array(data)
        assert len(data.shape) < 3 # invalid data shape        
        assert rot_type in RotType, "Invalid type string" # invalid type string
        
        if rot_type == RotType.SO3:
            if is_SO3(data) is False: 
                raise Exception(f'Invalid Shape Error. SO3 must be (n,3,3) or (3,3), but got {data.shape}')
            else: self.data = data
        elif rot_type == RotType.so3:
            if is_so3(data) is False:
                raise Exception(f'Invalid Shape Error. so3 must be (n,3) or (3), but got {data.shape}')
            self.data = so3_to_SO3(data) 
        elif rot_type == RotType.QUAT_XYZW or rot_type == RotType.QUAT_WXYZ:
            if is_quat(data) is False:
                raise Exception(f'Invalid Shape Error. Quaternion must be (n,4) or (4), but got {data.shape}')
            self.data = quat_to_SO3(data, rot_type==RotType.QUAT_XYZW)        
        elif rot_type == RotType.RPY:
            if is_rpy(data) is False:
                raise Exception(f'Invalid Shape Error. RPY must be (3,), but got {data.shape}')
            self.data = rpy_to_SO3(data)
        
    # constructor
    @staticmethod
    def from_mat3(mat3: Array):
        return Rotation(mat3, RotType.SO3)
    @staticmethod
    def from_so3(so3: Array):
        return Rotation(so3, RotType.so3)
    @staticmethod
    def from_quat_xyzw(quat: Array):
        return Rotation(quat, RotType.QUAT_XYZW)
    @staticmethod
    def from_quat_wxyz(quat: Array):
        return Rotation(quat, RotType.QUAT_WXYZ)
    @staticmethod
    def from_rpy(rpy: Array):
        return Rotation(rpy, RotType.RPY)
        
    @property
    def type(self):
        return type(self.data)
    
    @property
    def dtype(self):
        return self.dtype
    
    def mat(self):
        return self.data
    
    def so3(self):
        return SO3_to_so3(self.data)  
    
    def quat(self):
        return SO3_to_quat(self.data)
    
    def apply_pts3d(self, pts3d: Array):
        ## R*pts3d: [3,3] * [3,n]
        assert pts3d.shape[0] == 3, f"Invalid Shape. pts3d's shape should be (3,n), but got {str(pts3d.shape)}."
        mat = self.mat()
        if is_tensor(pts3d): mat = convert_tensor(mat,pts3d)        
        pts3d = matmul(mat,pts3d) # [3,3] * [3,n] = [3,n]
        # pts3d = transpose2d(pts3d).reshape(-1,3) # [n,3]
        return pts3d # [3,n]
    
    def inverse_mat(self)->Array:
        return transpose2d(self.data)
    
    def inverse(self) -> 'Rotation':
        return Rotation.from_mat3(self.inverse_mat())
    
    def _dot(self, rot:'Rotation') -> 'Rotation':
        rot1_mat = self.mat()
        rot2_mat = rot.mat()
        rot2_mat = convert_array(rot2_mat,rot1_mat)
        rot_mat = matmul(rot1_mat,rot2_mat)
        return Rotation.from_mat3(rot_mat)

    def __mul__(self, other: 'Rotation') -> 'Rotation':
        if not isinstance(other, Rotation):
            raise ValueError("Multiplication is only supported between Rotation instances")
        return self._dot(other)

def slerp(r1:Rotation,r2:Rotation, t:float):
    """
    Spherical Linear Interpolation between two Rotion
    1. transfrom Rotations to unit quaternions q1,q2
    2. compute angle "w" between two quaternions: w = cos^-1(q1*q2)
    3. compute slerp(q1,q2,t) =  sin((1-t)*w)/sin(w)*q1 + sin(tw)/sin(w)*q2

    Args:
        r1: Rotation, rotation instance
        r2: Rotation, rotation instance
        t: float, interploation parameters, 0<=t<=1
    Return:
        slerp(q1,q2,t): Rotation
    """
    assert (t <=1. and t >= 0.), ("Interpolation parameters must be between 0 and 1.")
    
    if t == 1.: return r2
    if t == 0.: return r1

    q1,q2 = r1.quat(),r2.quat() # (w,x,y,z)
    cos_omega = matmul(q1,q2)
    if cos_omega < 0.: # If negative dot product, negate one of the quaternions to take shorter arc
        q1 = -q1
        cos_omega = -cos_omega
    if cos_omega > 0.9999: # If the quaternions are very close, use linear interpolation
        q = normalize(q1*(1-t) + q2*t,dim=0)
    else:
        omega = arcos(cos_omega)
        sin_omega = sin(omega)
        scale1 = sin((1-t)*omega) / sin_omega
        scale2 = sin(t*omega) / sin_omega
        q = normalize(scale1*q1 + scale2*q2,dim=0)
    return Rotation.from_quat_wxyz(q)  