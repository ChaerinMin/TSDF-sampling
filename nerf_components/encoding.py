
# TODO: Integrated Postional Encoding, HashEncoding,  Spherical Harmonics
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import gin
from typing import Tuple, Dict
from extensions.hashencoder import HashEncoder
import tinycudann as tcnn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.out_dim:int = 0
        self.use_grid = False
    
    def forward(self, x:Tensor):
        raise NotImplementedError

@gin.configurable
class PositionalEncoder(Encoder):
    def __init__(self, num_freqs:int = 10,
                 include_input: bool = True,
                 log_sampling: bool = True,
                 enc_input: str = 'position',
                 input_dim: int = 3
                 ):
        super(PositionalEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.out_dim = 2 * num_freqs * input_dim + (input_dim if include_input else 0)
        self.enc_input = enc_input 
        
        coeffs = torch.pow(2.0, torch.arange(num_freqs)) if log_sampling else torch.linspace(1, 2.0 ** (num_freqs - 1), num_freqs)
        self.register_buffer("coeffs",coeffs)

    def forward(self, input_dict:Dict[str,Tensor]) -> Tensor:
        x = input_dict[self.enc_input]
        embedding = [x] if self.include_input else []
        for coeff in self.coeffs:
            embedding.append(torch.sin(x*coeff))
            embedding.append(torch.cos(x*coeff))
        return torch.cat(embedding,dim=-1)


@gin.configurable
class HashEncoderWithPositional(Encoder):
    def __init__(self, num_freqs:int = 6,
                 include_input: bool = True,
                 log_sampling: bool = True,
                 input_dim:int = 3, # HashGridEncoder 
                 num_levels:int = 16,
                 level_dim:int = 2,
                 per_level_scale:float = 2.,
                 base_resolution:int = 16,
                 log2_hashmap_size:int = 19,
                 desired_resolution:int = None,
                 enc_input:str = 'position'):
        super(HashEncoderWithPositional, self).__init__()
        
        self.pos_encoder = PositionalEncoder(num_freqs,include_input,log_sampling,enc_input,input_dim)
        self.hash_encoder = HashGridEncoder(input_dim,num_levels,level_dim,per_level_scale,base_resolution,log2_hashmap_size,desired_resolution,enc_input)
        self.out_dim = self.pos_encoder.out_dim + self.hash_encoder.out_dim
        self.enc_input = enc_input
        self.use_grid = True
        
    def forward(self, input_dict:Dict[str,Tensor]) -> Tensor:
        freq_enc_feature = self.pos_encoder(input_dict)
        hash_enc_feature = self.hash_encoder(input_dict)
        return torch.concat([hash_enc_feature,freq_enc_feature], -1)
    
    def grid_parameter(self):
        return  self.hash_encoder.grid_parameters()

@gin.configurable
class HashGridEncoder(Encoder):
    """
    HashGrid Encoder which Wrapping HashEncoder from MonoSDF:
    https://github.com/autonomousvision/monosdf/tree/main/code/hashencoders
    """
    def __init__(self,
                 input_dim:int = 3,
                 num_levels:int = 16,
                 level_dim:int = 2,
                 per_level_scale:float = 2.,
                 base_resolution:int = 16,
                 log2_hashmap_size:int = 19,
                 desired_resolution:int = None,
                 enc_input: str = 'position'
                 ):
        super(HashGridEncoder, self).__init__()
        self.hash_encoder = HashEncoder(
            input_dim,num_levels,level_dim,per_level_scale,base_resolution,log2_hashmap_size,desired_resolution)
        self.enc_input = enc_input
        self.out_dim = self.hash_encoder.output_dim
        self.use_grid = True

    def forward(self, input_dict:Dict[str,Tensor]) -> Tensor:
        input = input_dict[self.enc_input]
        return self.hash_encoder(input)
    
    def grid_parameters(self):
        return self.hash_encoder.parameters()

@gin.configurable
class SHEncoder(Encoder):
    def __init__(self,
                 input_dim:int = 3,
                 degree:int = 4,
                 enc_input: str = 'direction'):
        super(SHEncoder, self).__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.enc_input = enc_input
        
        self.tcnn_sh_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": self.degree,
                },
            )
        self.out_dim = self.tcnn_sh_encoder.n_output_dims
    def forward(self, input_dict:Dict[str,Tensor]) -> Tensor:
        input = input_dict[self.enc_input]
        return self.tcnn_sh_encoder(input)
    

if __name__ == "__main__":
    input = torch.tensor([[1,2,3]])
    pos_encoder = PositionalEncoder(num_freqs=2)
    print(pos_encoder(input))
