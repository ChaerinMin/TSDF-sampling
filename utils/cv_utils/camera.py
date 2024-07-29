import numpy as np

from constant import PI
from hybrid_operations import *
from hybrid_math import *
from enum import Enum
from typing import *
from scipy.ndimage import map_coordinates
import copy

class CamType(Enum):
    PINHOLE = ("PINHOLE", "Pinhole Camera Type")
    EQUIDISTANT = ("EQUIDISTANT", "Equidistant Camera Type")
    EQUIRECT = ("EQUIRECT", "Equirectangular Camera Type")
    NONE = ("NONE", "No Camera Type")
    
    @staticmethod
    def from_string(type_str: str)-> 'CamType':
        if type_str == 'PINHOLE':
            return CamType.PINHOLE
        elif type_str == 'OPENCV_FISHEYE':
            return CamType.EQUIDISTANT
        elif type_str == 'EQUIRECT':
            return CamType.EQUIRECT
        else:
            return CamType.NONE

# Camera Interface
class Camera:
    def __init__(self, cam_dict: Dict[str, Any]):
        self.cam_type = CamType.NONE
        self.cam_dict = cam_dict
        self.width,self.height = cam_dict['image_size']
    
    def make_pixel_grid(self) -> np.ndarray:
        u,v = np.meshgrid(range(self.width), range(self.height))
        uv = concat([u.reshape((1,-1)), v.reshape((1,-1))], 0)
        return uv 
    
    def get_rays(self) -> Array:
        raise NotImplementedError
    
    def project_rays(self, rays: Array, out_subpixel:bool=False) -> Array:
        raise NotImplementedError
    
    @property
    def resolution(self):
        return (self.width,self.height)
    
    def export_cam_dict(self):
        cam_dict = copy.deepcopy(self.cam_dict) 
        cam_dict["cam_type"] = self.cam_type.name
        return cam_dict
        
    @staticmethod
    def create_cam(cam_dict: Dict[str, Any]) -> 'Camera':
        cam_type = CamType.from_string(cam_dict['cam_type']) 
        
        if cam_type is CamType.PINHOLE:
            return PinholeCamera(cam_dict)
        elif cam_type is CamType.EQUIDISTANT:
            raise NotImplementedError
        elif cam_type is CamType.EQUIRECT:
            return EquirectangularCamera(cam_dict)
        else:
            raise Exception("Unkwon Camera Type")

# Abstract Class for Radial Camera Model
class RadialCamera(Camera):
    def __init__(self,cam_dict: Dict[str,Any]):
        super(RadialCamera,self).__init__(cam_dict)
        self.fx, self.fy = cam_dict['focal_length']
        self.cx, self.cy = cam_dict['principal_point']
        self.skew = cam_dict['skew']
    
    @property
    def K(self):
        K = np.eye(3)
        K[0,0] = self.fx
        K[1,1] = self.fy
        K[0,1] = self.skew
        K[0,2] = self.cx
        K[1,2] = self.cy
        return K
    
    @property
    def inv_K(self):
        return inv(self.K)
    
    @property
    def fov(self):
        fov_x = 2 * rad2deg(arctan(self.width / (2 * self.fx))) 
        fov_y = 2 * rad2deg(arctan(self.height / (2 * self.fy))) 
        return fov_x, fov_y

    def _to_image_plane(self, x:Array, y:Array, use_clip:bool=False) -> Tuple[Array,Array]:
        u = self.fx * x + self.skew * y + self.cx
        v = self.fy * y + self.cy
        if use_clip:
            u = clip(u,0.,float(self.width))
            v = clip(v,0.,float(self.height))
        return u,v
    
    def _to_normalized_plane(self, uv:Array) -> Tuple[Array,Array]:
        x = (uv[0:1,:] - self.cx - self.skew / self.fy*(uv[1:2,:] -self.cy)) / self.fx # (1,HW)
        y = (uv[1:2,:] - self.cy) / self.fy #(1,HW)
        return x,y

    def has_distortion(self):
        raise NotImplementedError

    def _distort(self, x: Array, y: Array, extra: bool=False) \
            -> Union[Tuple[Array,Array],Tuple[Array,Array,Array,Array]] :
        raise NotImplementedError    
        
    def _undistort(self, xd: Array, yd: Array) -> Tuple[Array,Array]:
        raise NotImplementedError        

    def _to_image_plane(self, x:Array, y:Array) -> Tuple[Array,Array]:
        u = self.fx * x + self.skew * y + self.cx
        v = self.fy * y + self.cy
        return u,v
    
    def _to_normalized_plane(self, uv:Array) -> Tuple[Array,Array]:
        x = (uv[0:1,:] - self.cx - self.skew / self.fy*(uv[1:2,:] -self.cy)) / self.fx # (1,HW)
        y = (uv[1:2,:] - self.cy) / self.fy #(1,HW)
        return x,y

    def get_rays(self,
                 uv:Array = None, 
                 norm:bool = True,
                 out_scale:bool = False
                 ) -> Array:
        if uv is None: uv = self.make_pixel_grid() # (2,HW)
        x,y = self._to_normalized_plane(uv) 
        if self.has_distortion():
            x,y = self._undistort(x,y)
        z = ones_like(x)
        rays = concat([x,y,z], 0) # (3,HW)
        if norm: rays = normalize(rays,dim=0)
        if out_scale:
            depth_scale = rays[2:3,:]
            return rays, depth_scale
        return rays # (3,HW)
    
    def project_rays(self, rays: Array, out_subpixel:bool=False) -> Array:
        X = rays[0:1,:]
        Y = rays[1:2,:]
        Z = rays[2:3,:]
        x,y = X / Z, Y / Z
        if self.has_distortion(): x,y = self._distort(x,y)
        u,v = self._to_image_plane(x,y)
        uv = concat([u,v], dim=0)
        return uv if out_subpixel else as_int(uv,n=32) # (2,HW)

# For Simple Radial Camera Model or small distortion
class PinholeCamera(RadialCamera):
    """
    Pinhole Camera Model.

    Attributes:
        cam_type (CamType): Camera type, set to PINHOLE.
        fx, fy (float): Focal length in x and y directions.
        cx, cy (float): Principal point coordinates (center of the image).
        skew (float): Skew coefficient between x and y axis.
        radial_params (List[float]): Radial distortion parameters [k1, k2, k3].
        tangential_params (List[float]): Tangential distortion parameters [p1, p2].
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super(PinholeCamera, self).__init__(cam_dict)
        self.cam_type = CamType.PINHOLE
        self.radial_params = cam_dict['radial']
        self.tangential_params = cam_dict['tangential']
        self.err_thr: float = 1e-4
        self.max_iter:int=5
    
    @property
    def dist_coeffs(self):
        _dist_coeffs = self.radial_params[0:2] + self.tangential_params + self.radial_params[2:3]
        return np.array(_dist_coeffs)
    
    @staticmethod
    def from_K(K: List[List[float]], image_size:List[int], dist: List[float] = None) -> 'PinholeCamera':
        """
        Static method to create a PinholeCamera instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[List[float]]): Intrinsic matrix parameters as a list 3*3 format.
            image_size (List[int]): Image resolution as a list [width, height].
            dist (List[float]): Distortion coefficients as a list [k1, k2, p1, p2, k3].

        Returns:
            PinholeCamera: An instance of PinholeCamera with given parameters.
        """

        cam_dict = {
            'image_size': image_size,
            'focal_length': (K[0][0], K[1][1]), # fx and fy
            'principal_point': (K[0][2], K[1][2]), # cx and cy
            'skew': K[0][1],
        }
        
        if dist is not None:
            cam_dict["radial"] = (dist[0],dist[1],dist[4]) # k1, k2, k3
            cam_dict["tangential"] = (dist[2],dist[3]) # p1, p2
        else:
            cam_dict["radial"] = [0.,0.,0.]
            cam_dict["tangential"] = [0.,0.]

        return PinholeCamera(cam_dict)
    
    @staticmethod
    def from_fov(image_size:List[int], fov:Union[List[float],float]):
        """
        Static method to create a PinholeCamera instance from image resolution and field of view.

        Args:
            image_size (List[int]): Image resolution as a list [width, height].
            fov(Union[List[float],float]): Field of view in degrees as a list [fov_x, fov_y] or width field of view.

        Returns:
            PinholeCamera: An instance of PinholeCamera with calculated parameters.
        """
        width,height = image_size
        if type(fov) == float:
            fov_x = fov
            fov_y = fov
        else:
            fov_x, fov_y = fov
        # Calculate focal length based on FOV and image resolution
        fx = width / (2 * tan(deg2rad(fov_x) / 2))
        fy = height / (2 * tan(deg2rad(fov_y) / 2))
        # Assuming principal point is at the center of the image
        cx = width / 2.
        cy = height / 2.
        # Assuming no skew and no distortion
        skew = 0.
        radial_params = [0, 0, 0]  # No radial distortion
        tangential_params = [0, 0]  # No tangential distortion

        cam_dict = {
            'image_size': image_size,
            'focal_length': (fx, fy),
            'principal_point': (cx, cy),
            'skew': skew,
            'radial': radial_params,
            'tangential': tangential_params
        }
        return PinholeCamera(cam_dict)

    def has_distortion(self):
        cnt = np.count_nonzero(self.radial_params) + np.count_nonzero(self.tangential_params)
        return cnt != 0

    def _distort(self, x: Array, y: Array, extra: bool=False) \
            -> Union[Tuple[Array,Array],Tuple[Array,Array,Array,Array]] :
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        xy = x*y
        k1,k2,k3 = self.radial_params 
        rd = 1. + r2 * (k1 + r2 * (k2 + k3 *r2))
        # Tangential distortion term (tdx,tdy)
        p1, p2 = self.tangential_params
        tdx = 2*p1*xy + p2*(r2 + 2*x2) 
        tdy = p1*(r2 + 2*y2) + 2*p2*xy
        xd = x * rd + tdx
        yd = y * rd + tdy
        
        if extra:
            # Compute derivative of radial distortion, x, y
            rd_r = k1 + r2 * (2. * k2 + 3. * k3 * r2)
            rd_x = 2. * x * rd_r
            rd_y = 2. * y * rd_r
            return xd, yd, rd, rd_x, rd_y
        return xd, yd
        
    def _compute_residual_jacobian(self,xu:Array,yu:Array, xd:Array, yd:Array) \
            -> Tuple[Array,Array,Array,Array,Array,Array]:
        
        """
        Computes the Jacobian matrix of the residual between undistorted (ideal) points and distorted (actual) points
        caused by camera lens distortion. Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py    
        Parameters:
        - xu, yu: Arrays of x and y coordinates of the undistorted points.
        - xd, yd: Arrays of x and y coordinates of the distorted points.
    
        Returns:
        - Tuple containing six arrays:
        - x_res, y_res: The residuals in x and y directions.
        - x_res_x, x_res_y: Partial derivatives of the x residual with respect to x and y.
        - y_res_x, y_res_y: Partial derivatives of the y residual with respect to x and y.

        The derivatives are calculated based on the distortion model which includes radial and tangential components.
        """
        # Compute distorted coordinates (_xd, _yd) and the distortion terms
        # Radial distortio term and its partial derivative (d, d_x, d_y)
        _xd,_yd, d, d_x, d_y = self._distort(xu,yu,True)
        x_res,y_res = _xd - xd, _yd - yd
        
        p1, p2 = self.tangential_params[0], self.tangential_params[1]
        # Compute derivative of x_dist over x and y 
        x_res_x = d + d_x * xu + 2. * p1 * yu + 6. * p2 * xu
        x_res_y = d_y * xu + 2. * p1 * xu + 2. * p2 * yu
        # Compute derivative of y_dist over x and y        
        y_res_x = d_x * yu + 2.0 * p2 * yu + 2.0 * p1 * xu
        y_res_y = d + d_y * yu + 2.0 * p2 * xu + 6.0 * p1 * yu
        
        return x_res, y_res, x_res_x, x_res_y, y_res_x, y_res_y
        
    def _undistort(self, xd: Array, yd: Array) -> Tuple[Array,Array]:
        """
        Compute undistorted coords
        Adapted from MultiNeRF
        https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509
            
        Args:
            xd: (1,N), float, The distorted coordinates x.
            yd: (1,N), float, The distorted coordinates y.
            err_thr: scalar, float, error threshold.
            max_iter: scalar, int, maximum iteration times.        
        Returns:
            xu: (1,N), float, The undistorted coordinates x.
            yu: (1,N), float, The undistorted coordinates y.
            
        We want to undistortion point like distortion == distort(undistortion)
        x,y := undistorted coords. x,y
        r(x,y):= (rx,ry) = distort(x,y) - (xd,yd), residual of undistorted coords.
        J(x,y) := |rx_x rx_y|  , Jacobian of residual
                  |ry_x ry_y|
        J^-1 = 1/D | ry_y -rx_y|
                   |-ry_x  rx_x| 
        D := rx_x * ry_y - rx_y * ry_x, Determinant of Jacobian
        
        Initialization:
            x,y := xd,yd
        Using Newton's Method:
            Iteratively
             x,y <- x,y - J ^-1 * (rx ry)^T  = (x,y) - 1/D * (ry_y*rx-rx_y*ry,rx_x*ry-ry_x*rx)   
        """
        xu,yu = deep_copy(xd), deep_copy(yd)
        
        for _ in range(self.max_iter):
            rx, ry, rx_x, rx_y, ry_x, ry_y \
                = self._compute_residual_jacobian(xu, yu, xd, yd)
            if sqrt(rx**2 + ry**2).max() < 1e-4: break
            Det = ry_x * rx_y - rx_x * ry_y
            x_numerator,y_numerator = ry_y*rx-rx_y*ry, rx_x*ry-ry_x*rx
            step_x = where(abs(Det) > self.err_thr, x_numerator / Det, zeros_like(Det))
            step_y = where(abs(Det) > self.err_thr, y_numerator / Det, zeros_like(Det))
            xu = xu + clip(step_x,-0.5, 0.5)
            yu = yu + clip(step_y,-0.5, 0.5)

        return xu,yu

# For OpenCV Fisheye Camera Model
class EquidistantCamera(RadialCamera):
    """
    Represents an Equidistant (Equiangular) Camera Model, typically used for wide-angle or fisheye lenses.
    This model is characterized by its equidistant projection, where radial distances are proportional to the actual angles.

    Attributes:
        cam_type (CamType): Camera type, set to EQUIDISTANT, indicating the equidistant projection model.
        fx, fy (float): Focal length in x and y directions. These parameters define the scale of the image on the sensor.
        cx, cy (float): Principal point coordinates (center of the image), indicating where the optical axis intersects the image sensor.
        skew (float): Skew coefficient between x and y axis, representing the non-orthogonality between these axes.
        radial_params (List[float]): Radial distortion parameters [k1, k2, k3, k4], specifying the lens's radial distortion. Typically used in distortion correction algorithms.
    
    https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#ga75d8877a98e38d0b29b6892c5f8d7765    
    """

    def __init__(self, cam_dict: Dict[str, Any]):
        super(EquidistantCamera, self).__init__(cam_dict)      
        self.cam_type = CamType.EQUIDISTANT
        self.radial_params = cam_dict['radial'] # k1,k2,k3,k4
        self.err_thr: float = 1e-4
        self.max_iter:int=20
        
        # Polynomial coefficients for fast undistortion
        if "poly_coeffs" in cam_dict:
            self.poly_coeffs = cam_dict["poly_coeffs"]
        else:
            self.poly_coeffs = self._compute_polynomial_coeffs()
      
    @property
    def dist_coeffs(self):
        return np.array(self.radial_params) 

    @staticmethod
    def from_K_D(K: List[float], image_size:List[int], D: List[float]) -> 'EquidistantCamera':
        """
        Static method to create an EquidistantCamera (fisheye) instance from intrinsic matrix and distortion coefficients.

        Args:
            K (List[float]): Intrinsic matrix parameters as a list [fx, skew, cx, fy, cy].
            image_size (List[int]): Image resolution as a list [width, height].
            D (List[float]): Distortion coefficients as a list [k1, k2, k3, k4].
        Returns:
            EquidistantCamera: An instance of EquidistantCamera with given parameters.
        """

        cam_dict = {
            'image_size': image_size,
            'focal_length': (K[0][0], K[1][1]), # fx and fy
            'principal_point': (K[0][2], K[1][2]), # cx and cy
            'skew': K[0][1],
            'radial': D
        }
        return EquidistantCamera(cam_dict)
    
    def has_distortion(self):
        return True
    
    def _distort(self, x: Array, y: Array, extra:bool=False) \
            -> Tuple[Array,Array]:
        x2, y2 = x**2, y**2
        r2 = x2 + y2

        k1,k2,k3,k4 = self.radial_params 
        rd = 1. + r2 * (k1 + r2 * (k2 + r2* (k3 + r2 * k4)))
        xd = x * rd
        yd = y * rd

        if extra:
            rd_r =  k1 + r2 * (2. * k2 + r2 * (3 * k3 + 4. * k4 * r2))
            rd_x = 2. * x * rd_r 
            rd_y = 2. * y * rd_r 
            return xd, yd, rd, rd_x, rd_y
        return xd, yd

    def _compute_polynomial_coeffs(self, degree: int = 5) -> Array:
        uv = self.make_pixel_grid()
        x, y = self._to_normalized_plane(uv)

        # Apply distortion
        xd, yd = self._distort(x, y)

        # Fit polynomials
        coeffs_x = polyfit(xd.flatten(), x.flatten(), degree)
        coeffs_y = polyfit(yd.flatten(), y.flatten(), degree)
        
        return coeffs_x, coeffs_y

    def _compute_residual_jacobian(self,xu:Array,yu:Array, xd:Array, yd:Array) \
            -> Tuple[Array,Array,Array,Array,Array,Array]:
        
        """
        Computes the Jacobian matrix of the residual between undistorted (ideal) points and distorted (actual) points
        caused by camera lens distortion. Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py    
        Parameters:
        - xu, yu: Arrays of x and y coordinates of the undistorted points.
        - xd, yd: Arrays of x and y coordinates of the distorted points.
    
        Returns:
        - Tuple containing six arrays:
        - x_res, y_res: The residuals in x and y directions.
        - x_res_x, x_res_y: Partial derivatives of the x residual with respect to x and y.
        - y_res_x, y_res_y: Partial derivatives of the y residual with respect to x and y.

        The derivatives are calculated based on the radial distortion model.
        """
        # Compute distorted coordinates (_xd, _yd) and the distortion terms
        # Radial distortion term and its partial derivative (d, d_x, d_y)        
        _xd,_yd, d, d_x, d_y = self._distort(xu,yu,True)
        x_res,y_res = _xd - xd, _yd - yd
        
        # Compute derivative of x_dist over x and y 
        x_res_x = d + d_x * xu 
        x_res_y = d_y * xu
        # Compute derivative of y_dist over x and y        
        y_res_x = d_x * yu
        y_res_y = d + d_y * yu
        
        return x_res, y_res, x_res_x, x_res_y, y_res_x, y_res_y

    def _undistort(self, xd:Array, yd:Array) \
        -> Tuple[Array,Array]:
        
        if self.poly_coeffs is not None:
            # Evaluate polynomials
            xu = polyval(self.poly_coeffs[0], xd)
            yu = polyval(self.poly_coeffs[1], yd)
        else:        
            xu,yu = deep_copy(xd), deep_copy(yd)
            for _ in range(self.max_iter):
                rx, ry, rx_x, rx_y, ry_x, ry_y \
                    = self._compute_residual_jacobian(xu, yu, xd, yd)
                if sqrt(rx**2 + ry**2).max() < 1e-4: break
                Det = ry_x * rx_y - rx_x * ry_y
                x_numerator,y_numerator = ry_y*rx-rx_y*ry, rx_x*ry-ry_x*rx
                step_x = where(abs(Det) > self.err_thr, x_numerator / Det, zeros_like(Det))
                step_y = where(abs(Det) > self.err_thr, y_numerator / Det, zeros_like(Det))
                xu = xu + clip(step_x,-0.5, 0.5)
                yu = yu + clip(step_y,-0.5, 0.5)
        return xu,yu

class EquirectangularCamera(Camera):
    """
    Equirectangular Camera Model.
    
    This camera model is used for representing 360-degree panoramic images
    using equirectangular projection. It maps spherical coordinates to a 2D
    equirectangular plane.

    Attributes:
        cam_type (CamType): Camera type, set to EQUIRECT (Equirectangular).
        min_phi_deg (float): Minimum vertical field of view angle (phi) in degrees.
        max_phi_deg (float): Maximum vertical field of view angle (phi) in degrees.
        cx (float): The x-coordinate of the central point of the equirectangular image.
        cy (float): The y-coordinate of the central point of the equirectangular image.
    """
    def __init__(self, cam_dict: Dict[str, Any]):
        """
        Initializes the EquirectangularCamera with camera parameters.

        Args:
            cam_dict (Dict[str, Any]): A dictionary containing camera parameters.
                Expected to have 'min_phi_deg' and 'max_phi_deg' as keys with corresponding values.
        """
        super(EquirectangularCamera, self).__init__(cam_dict)
        self.cam_type = CamType.EQUIRECT
        
        self.min_phi_deg:float = cam_dict["min_phi_deg"]
        self.max_phi_deg:float = cam_dict["max_phi_deg"]
        self.cx = (self.width - 1.) / 2.0
        self.cy = (self.height - 1.) / 2.0

        self.phi_scale = deg2rad(self.max_phi_deg - self.min_phi_deg)
        self.phi_offset = deg2rad((self.max_phi_deg + self.min_phi_deg)*0.5)

    def get_rays(self, uv:np.ndarray = None, out_scale:bool=False):
        if uv is None: uv = self.make_pixel_grid()
        theta = (uv[0:1,:]-self.cx) / self.width * PI * 2
        phi = (uv[1:2,:] - self.cy) / self.height * self.phi_scale + self.phi_offset
        x = sin(theta) * cos(phi)
        y = sin(phi)
        z = cos(theta) * cos(phi)
        ray = concat([x,y,z],0)
        invalid_ray = logical_or(phi < self.min_phi_deg, phi > self.max_phi_deg).reshape(-1,)
        ray[:,invalid_ray] = np.nan if is_numpy(ray) else torch.nan
        if out_scale:
            depth_scale = ones_like(z)
            depth_scale[:,invalid_ray] = np.nan if is_numpy(ray) else torch.nan
            return ray, depth_scale
        return ray # (3,HW)
    
    def project_rays(self, rays:Array) -> Array:
        # Normalize the ray vector
        rays = normalize(rays, dim=0)
        X = rays[0:1,:]
        Y = rays[1:2,:]
        Z = rays[2:3,:]
        
        # Convert Cartesian coordinates to spherical coordinates
        theta = arctan2(X,Z)
        phi = arcsin(Y)

        # Convert spherical coordinates to pixel coordinates
        u = theta / (PI * 2) * self.width  + self.cx
        v = (phi - self.phi_offset) * self.height / self.phi_scale + self.cy
        uv = concat([u,v],dim=0)
        return as_int(uv,n=32) 