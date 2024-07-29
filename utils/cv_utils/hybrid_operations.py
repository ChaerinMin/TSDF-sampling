import numpy as np
from numpy import ndarray
from torch import Tensor
import torch
from typing import *

Array = Union[ndarray, Tensor] # Hybrid ArrayType

def is_tensor(x: Array) -> bool:
    return type(x) == Tensor

def is_numpy(x: Array) -> bool:
    return type(x) == ndarray

def is_array(x: Any) -> bool:
    return is_tensor(x) or is_numpy(x)

def convert_tensor(x: Array, tensor: Tensor = None) -> Tensor:
    if is_tensor(x): return x 
    if tensor is not None:    
        assert is_tensor(tensor)
        x_tensor = torch.tensor(x, dtype=tensor.dtype, device=tensor.device)
    else:
        x_tensor = Tensor(x)
    return x_tensor
        
def convert_numpy(x: Array) -> ndarray:  
    if is_tensor(x): x_numpy = x.detach().cpu().numpy()
    elif is_numpy(x): x_numpy = x
    else: x_numpy = np.array(x)
    return x_numpy

def convert_array(x:Any, array:Array) -> Array:
    if is_tensor(array): return convert_tensor(x,array)
    return convert_numpy(x)

def _assert_same_array_type(arrays: Tuple[Array, ...]):
    assert all(is_tensor(arr) for arr in arrays) or all(is_numpy(arr) for arr in arrays), "All input arrays must be of the same type"

def convert_dict_tensor(dict: Dict[Any, ndarray], tensor: Tensor=None) -> Dict[Any,Tensor]:
    # _assert_same_array_type(dict)
    new_dict = {}
    for key in dict.keys():
        new_dict[key] = convert_tensor(dict[key], tensor)
    return new_dict

def expand_dim(x: Array, dim: int) -> Array:
    if is_tensor(x): return x.unsqueeze(dim)
    else: return np.expand_dims(x, axis=dim)

def reduce_dim(x: Array, dim: int) -> Array:
    if is_tensor(x): return x.squeeze(dim)
    else: return np.squeeze(x, axis=dim)

def concat(x: List[Array], dim: int) -> Array:
    _assert_same_array_type(x)
    if is_tensor(x[0]): return torch.cat(x, dim=dim)
    return np.concatenate(x, axis=dim)

def stack(x: List[Array], dim:int) -> Array:
    _assert_same_array_type(x)
    if is_tensor(x[0]): return torch.stack(x, dim=dim)
    return np.stack(x, axis=dim)
  
def ones_like(x: Array) -> Array:
    assert is_array(x), ("Invalid Type. It is neither Numpy nor Tensor.")
    if is_tensor(x): return torch.ones_like(x)
    return np.ones_like(x)

def zeros_like(x: Array) -> Array:
    if is_tensor(x): return torch.zeros_like(x)
    return np.zeros_like(x)

def full_like(x:Array, fill_value:Any) -> Array:
    if is_tensor(x): torch.full_like(x,fill_value)
    else: np.full_like(x,fill_value)

def deep_copy(x:Array) -> Array:
    if is_tensor(x): return x.clone()
    return np.copy(x)

def where(condition:Array, x:Array, y:Array) -> Array:
    if is_tensor(condition): return torch.where(condition,x,y)
    return np.where(condition,x,y)

def clip(x:Array, min:float=None,max:float=None) -> Array:
    if is_tensor(x): return torch.clip(x,min,max)
    return np.clip(x,min,max)

def eye(n:int, x: Array) -> Array:
    if is_tensor(x): return torch.eye(n)
    return np.eye(n)

def as_int(x:Array,n:int=32) -> Array:
    if is_tensor(x):
        if n == 64: return x.type(torch.int64)
        elif n == 32: return x.type(torch.int32)
        elif n == 16: return x.type(torch.int16)
        else: raise TypeError
    elif is_numpy(x):
        if n == 256: return x.astype(np.int256)
        elif n == 128: return x.astype(np.int128)
        elif n == 64: return x.astype(np.int64)
        elif n == 32: return x.astype(np.int32)
        elif n == 16: return x.astype(np.int16)
        else: raise TypeError

def as_float(x:Array, n:int=32) -> Array:
    if is_tensor(x):
        if   n == 64: return x.type(torch.float64)
        elif n == 32: return x.type(torch.float32)
        elif n == 16: return x.type(torch.float16)
        else: raise TypeError
    elif is_numpy(x):
        if   n == 256:return x.astype(np.float256)
        elif n == 128:return x.astype(np.float128)
        elif n == 64: return x.astype(np.float64)
        elif n == 32: return x.astype(np.float32)
        elif n == 16: return x.astype(np.float16)
        else: raise TypeError

def logical_or(*arrays: Array) -> Array:
    assert len(arrays) > 0, "At least one input array is required"
    _assert_same_array_type(arrays)

    result = arrays[0]
    for arr in arrays[1:]:
        if is_tensor(result):
            result = torch.logical_or(result, arr)
        else:
            result = np.logical_or(result, arr)

    return result

def logical_and(*arrays: Array) -> Array:
    assert len(arrays) > 0, "At least one input array is required"
    _assert_same_array_type(arrays)

    result = arrays[0]
    for arr in arrays[1:]:
        if is_tensor(result):
            result = torch.logical_and(result, arr)
        else:
            result = np.logical_and(result, arr)
    return result

def logical_not(x: Array) -> Array:
    assert(is_array(x))
    if is_tensor(x): return torch.logical_not(x)
    return np.logical_not(x)

def logical_xor(x: Array) -> Array:
    assert(is_array(x))
    if is_tensor(x): return torch.logical_xor(x)
    return np.logical_xor(x)