import numpy as np
from typing import *
import tifffile
import skimage.io as skio
from skimage import img_as_float
import json
import yaml
from PIL import Image
import os

def read_npy(path: str) -> np.ndarray:
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Reading npy file {path} failed.")
        print(e)
        data = None
    return data

def read_tiff(path: str) -> np.ndarray:
    try:
        multi_datas = tifffile.TiffFile(path)
        num_datas = len(multi_datas.pages)
        if num_datas == 0:
            raise Exception("No Images.")
        elif num_datas == 1:
            data = multi_datas.pages[0].asarray().squeeze()
        else:
            data = np.concatenate([np.expand_dims(x.asarray(), 0) for x in multi_datas.pages], 0)
    except Exception as e:
        print(f"Reading tiff file {path} failed.")
        print(e)
        data = None
    return data

def write_npy(data: np.ndarray, path: str):
    try:
        np.save(path, data)
    except Exception as e:
        print(f"Writing npy file {path} failed.")
        print(e)

def write_tiff(data: np.ndarray, path: str):
    try:
        with tifffile.TiffWriter(path) as tiff:
            if data.dtype is not np.float32:
                data = data.astype(np.float32)
            tiff.write(data, photometric='MINISBLACK', bitspersample=32, compression='zlib')
    except Exception as e:
        print(f"Writing tiff file {path} failed.")
        print(e)

def read_all_images(image_dir: str, as_float: bool = False) -> Optional[List[np.ndarray]]:
    image_list = os.listdir(image_dir)
    images = []
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image = read_image(image_path, as_float)
        if image is None:
            return None
        images.append(image)
    return images 

def read_image(path: str, as_float:bool=False) -> np.ndarray:
    try:
        image = skio.imread(path)
        if as_float:
             image = img_as_float(image) # normalize [0,255] -> [0.,1.]
        return image 
    except Exception as e:
        print("reading image failed.")
        return None

def write_image(image:np.ndarray, path: str):
    if len(image.shape) == 3 and image.shape[-1]== 1:
        image = np.squeeze(image)
    pil_image = Image.fromarray(image)
    extension = path.split(sep=".")[-1]
    pil_image.save(path, extension)

def read_json(path: str) -> Dict[str,Any]:
    try:
        with open(path, "r") as f:
            json_dict = json.load(f)
        return json_dict 
    except:
        raise Exception("Reading json file Failed.")

def write_json(path:str, dict:Dict[str,Any], indent:int=1):
    try:
        with open(path,"w") as f:
            json.dump(dict,f, indent=indent)
    except:
        raise Exception("Writing json file Failed.")

def read_yaml(path:str) -> Dict[str, Any]:
    try:
        with open(path,"r") as f:
            dict = yaml.load(f,Loader=yaml.FullLoader)
        return dict
    except:
        raise Exception("Reading yaml file failed.")

def write_yaml(path:str, dict:Dict[str,Any]):
    with open(path,"w") as f:
        yaml.dump(dict,f)
