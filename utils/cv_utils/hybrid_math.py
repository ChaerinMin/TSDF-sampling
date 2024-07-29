from hybrid_operations import *
from scipy.linalg import expm
from constant import EPSILON

# Basic Mathematical Operations
def abs(x: Array) -> Array:
    if is_tensor(x): return torch.abs(x)
    return np.abs(x)

def sqrt(x: Array) -> Array:
    if is_tensor(x): return torch.sqrt(x)
    return np.sqrt(x)

def mean(x: Array, dim: int = None) -> Array:
    if dim is not None:
        assert dim < x.ndim, f"Invalid dimension: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.mean(x, dim=dim)
    return np.mean(x, axis=dim)

def dot(x: Array, y: Array) -> Array:
    assert(type(x) == type(y)), f"Invalid type: expected same type for both arrays, but got {type(x)} and {type(y)}"
    if is_tensor(x): return torch.dot(x, y)
    return np.dot(x, y)

def svd(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for SVD: expected a 2D matrix, but got {x.shape}."
    if is_tensor(x): return torch.linalg.svd(x)
    return np.linalg.svd(x)

def determinant(x: Array) -> float:
    assert x.ndim == 2, f"Invalid shape for determinant: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x):
        return torch.det(x).item()
    return np.linalg.det(x)

def inv(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for inversion: expected a 2D square matrix, but got {x.shape}."
    if is_tensor(x): return torch.inverse(x)
    return np.linalg.inv(x)

# Matrix and Vector Operations
def norm(x: Array, ord: Union[int, str] = None, dim: int = None, keepdim: bool = False) -> Array:
    if dim is not None and dim >= x.ndim:
        assert False, f"Invalid dimension for norm: expected dimension less than {x.ndim}, but got {dim}."
    if is_tensor(x): return torch.norm(x, p=ord, dim=dim, keepdim=keepdim)
    return np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim)

def normalize(x: Array, ord: Union[int, str] = None, dim: int = None, eps: float = EPSILON) -> Array:
    n = norm(x=x, ord=ord, dim=dim, keepdim=False)
    return x / (n + eps)

def transpose2d(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for transpose: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return x.transpose(0, 1)
    return x.T

def matmul(x: Array, y: Array) -> Array:
    assert x.ndim >= 1 and y.ndim >= 1, f"Invalid shape: expected at least 1D arrays, but got {x.shape} and {y.shape}."
    assert x.shape[-1] == y.shape[0], f"Invalid shape for matmul: x dimensions {x.shape[-1]}, y dimensions {y.shape[0]}."
    if is_tensor(x): return torch.matmul(x, y)
    return x @ y

def permute(x: Array, dims: Tuple[int]) -> Array:
    assert len(x.shape) == len(dims), f"Invalid permutation: expected dimensions {len(x.shape)}, but got {len(dims)}."
    if is_tensor(x): return x.permute(dims)
    return x.transpose(dims)

def trace(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for trace: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return torch.trace(x)
    return np.trace(x)

def vec3_to_skew(x: Array) -> Array:
    assert (x.shape == (3,) or x.shape == (1,3)), f"Invalid shape. Shape of vector must be (3,) or (1,3), but got {str(x.shape)}"
    if x.shape == (1,3): x = reduce_dim(x,0)
    wx = x[0].item()
    wy = x[1].item()
    wz = x[2].item()
    skew_x = np.array([[0., -wz, wy],
                    [wz,  0, -wx],
                    [-wy, wx,  0]])
    if is_tensor(x): skew_x = convert_tensor(skew_x, x)
    return skew_x

def diag(x: Array) -> Array:
    assert x.ndim == 2, f"Invalid shape for diag: expected a 2D array, but got {x.shape}."
    if is_tensor(x): return torch.diag(x)
    return np.diag(x)

# Transform and Permutation
def rad2deg(x: Array) -> Array:
    return x * (180. / np.pi)

def deg2rad(x: Array) -> Array:
    return x * (np.pi / 180.)

def exponential_map(mat: Array) -> Array:
    if is_tensor(mat):
        return torch.matrix_exp(mat)
    else:
        return expm(mat)

# Trigonometric functions
def sin(x: Array) -> Array:
    if is_tensor(x): return torch.sin(x)
    return np.sin(x)

def cos(x: Array) -> Array:
    if is_tensor(x): return torch.cos(x)
    return np.cos(x)

def tan(x: Array) -> Array:
    if is_tensor(x): return torch.tan(x)
    return np.tan(x)

def arcsin(x: Array) -> Array:
    if is_tensor(x): return torch.arcsin(x)
    return np.arcsin(x)

def arcos(x: Array) -> Array:
    if is_tensor(x): return torch.arccos(x)
    return np.arccos(x)

def arctan(x: Array) -> Array:
    if is_tensor(x): return torch.arctan(x)
    return np.arctan(x)

def arctan2(x: Array, y: Array) -> Array:
    if is_tensor(x):
        y = convert_tensor(y,x)
        return torch.arctan2(x, y)
    return np.arctan2(x,y)

# Polynomial functions
def polyval(coeffs: Array, x: Array) -> Array:
    y = zeros_like(x)
    for c in coeffs:
        y = y * x + c
    return y

def polyfit(x: Array, y: Array, degree: int) -> Array:
    if is_tensor(x):
        x_np = convert_numpy(x)
        y_np =  convert_numpy(y)
        coeffs_np = np.polyfit(x_np, y_np, degree)
        return convert_tensor(coeffs_np,x)
    else:
        return np.polyfit(x, y, degree)

# Linear-Algebra Problem
def is_square(x: Array) -> bool:
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def solve(A:Array, b: Array) -> Array:
    assert(type(A) == type(b)), "Two Array must be same type."
    if is_tensor(A): return torch.linalg.solve(A,b)
    return np.linalg.solve(A,b)

def solve_linear_system(A:Array, b: Array = None):
    """
    Solve the linear system Ax = b or find the null space if b is None.
    Efficient for small linear systems but may be adapted for larger systems with appropriate libraries.

    Args:
    A: Array or Tensor representing the coefficient matrix.
    b: Optional; Array or Tensor representing the dependent variable vector. If None, find null space of A.

    Returns:
    Array or Tensor: Solution vector or null space basis vectors.
    """
    if b is not None:
        # Ax = b 
        return solve(A,b)    
    else:
        # Ax = 0
        # use svd
        _,s,vt = svd(A)
        null_space = transpose2d(vt)[:, s < EPSILON]
        return null_space