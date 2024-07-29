import torch
import torch.nn as nn
from nerf_components.encoding import *
from nerf_components.mlp import * 
from nerf_components.density import *
from nerf_components.field import * 
import gin
from typing import Dict,Tuple
from dataclasses import dataclass

"""
BaseNetwork: num samples are same each rays 
"""

@gin.configurable
def build_network_fn(geometric_field_fn:build_field_fn,color_field_fn:build_field_fn):
    """
    Build and return a BaseNetwork instance using configurations provided by gin.
    This function relies on gin-config to automatically inject dependencies when creating the BaseNetwork. 
    The exact configurations required by this function should be defined in the gin configuration file.
    Returns:
        BaseNetwork: An instance of BaseNetwork initialized with gin-config specified parameters.
    """
    network = BaseNetwork(geometric_field_fn(),color_field_fn())
    return network

class BaseNetwork(nn.Module):
    def __init__(self,
                geometric_field:GeometricField,
                color_field:ColorField):
        super(BaseNetwork,self).__init__()

        self.geometric_field = geometric_field
        self.color_field = color_field
    
    def forward(self, input_dict:Dict[str,Tensor]) -> Dict[str,Tensor]:
        geo_input = {}
        for component in self.geometric_field.input_components:
            if component in input_dict: geo_input[component] = input_dict[component]
        geo_input[self.geometric_field.encoder.enc_input] = input_dict[self.geometric_field.encoder.enc_input]
        geo_output = self.geometric_field(geo_input)
        color_input = {}
        for component in self.color_field.input_components:
            if component in input_dict: color_input[component] = input_dict[component]
            elif component in geo_output: color_input[component] = geo_output[component]
        color_input[self.color_field.encoder.enc_input] = input_dict[self.color_field.encoder.enc_input]
        color_output = self.color_field(color_input)
        output_dict = {**geo_output, **color_output}
        return output_dict

    def get_density(self, input_dict:Dict[str,Tensor]):
        geo_input = {}
        for component in self.geometric_field.input_components:
            if component in input_dict: geo_input[component] = input_dict[component]
        geo_input[self.geometric_field.encoder.enc_input] = input_dict[self.geometric_field.encoder.enc_input]
        return self.geometric_field.get_density(geo_input)

    def mlp_parameters(self):
        parameters = []
        parameters += list(self.geometric_field.mlp.parameters())
        parameters += list(self.color_field.mlp.parameters())
        return parameters
        