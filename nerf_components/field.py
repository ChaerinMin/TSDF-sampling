import torch
import torch.nn as nn
from nerf_components.encoding import *
from nerf_components.mlp import * 
from nerf_components.density import * 
import gin
from typing import Dict,Tuple


@gin.configurable
def build_field_fn(field_cls:'BaseField',
                   encoder_cls:Encoder,mlp_cls:VanillaMLP,input_components:List[str]) -> 'BaseField':
    return field_cls(encoder_cls,mlp_cls,input_components)

@gin.configurable
class BaseField(nn.Module):
    def __init__(self,
                 encoder_cls:Encoder,
                 mlp_cls:VanillaMLP,
                 input_components: Dict[str,int],
                 ):
        super(BaseField,self).__init__()
        self.input_components = input_components
        self.encoder = encoder_cls()
        input_dim = 0
        for comp in input_components:
            if comp == 'enc_feature':
                input_dim += self.encoder.out_dim
            else:
                input_dim += input_components[comp]
        self.mlp = mlp_cls(input_dim=input_dim)    
        
    def _concat_input(self,input_dict:Dict[str,Tensor]):
        return  torch.cat([input_dict[component] for component in self.input_components], dim=-1) # B * (D1 + D2 + ...)
    
    def _encoding(self, input_dict: Dict[str,Tensor]) -> Tensor:
        return self.encoder(input_dict)
        
    def forward(self, input_dict: Dict[str,Tensor]) -> Tensor:
        input_dict["enc_feature"] = self._encoding(input_dict)
        input = self._concat_input(input_dict)
        return self.mlp(input)

@gin.configurable
class NeRFDensityField(BaseField):
    def __init__(self,
                 encoder_cls:Encoder,
                 mlp_cls:VanillaMLP,
                 input_components: Dict[str,int]
                 ):
        super(NeRFDensityField,self).__init__(encoder_cls,mlp_cls,input_components)
        output_dim = self.mlp.output_dim
        self.alpha_linear = mlp_cls(input_dim=output_dim,num_hidden_layers=0,
                                    output_dim=1,activation=None,out_activation='ReLU')
        self.feature_linear = mlp_cls(input_dim=output_dim,num_hidden_layers=0,
                                 output_dim=output_dim,activation=None,out_activation=None)
        
    def forward(self,input_dict:Dict[str,Tensor]) -> Dict[str,Tensor]:
        output_dict = {}
        output = super().forward(input_dict)
        output_dict["density"] = self.alpha_linear(output)
        output_dict["geometric_feature"] = self.feature_linear(output)
        return output_dict
    
    def get_density(self, input_dict:Dict[str,Tensor]):
        output = super().forward(input_dict)
        return self.alpha_linear(output)

@gin.configurable
class GeometricField(BaseField):
    def __init__(self,
                 encoder_cls:Encoder,
                 mlp_cls:VanillaMLP,
                 input_components: Dict[str,int],
                 density_act_str:str
                 ):
        super(GeometricField,self).__init__(encoder_cls,mlp_cls,input_components)
        self.density_act_fn = activation_fn(density_act_str)
        
    def forward(self,input_dict:Dict[str,Tensor]) -> Dict[str,Tensor]:
        output_dict = {}
        output = super().forward(input_dict)
        output_dict["density"] = self.density_act_fn(output[:,0:1]) 
        output_dict["geometric_feature"] = output[:,1:]
        return output_dict        

    def get_density(self, input_dict:Dict[str,Tensor]) -> Tensor:
        return super().forward(input_dict)[:,0:1]
    
@gin.configurable
class NeRFColorField(BaseField):
    def __init__(self,
                 encoder_cls:Encoder,
                 mlp_cls:VanillaMLP,
                 input_components: Dict[str,int]
                 ):
        super(NeRFColorField,self).__init__(encoder_cls,mlp_cls,input_components)
        output_dim = self.mlp.output_dim
        self.color_linear = mlp_cls(input_dim=output_dim,num_hidden_layers=0,
                                    output_dim=3,activation=None,out_activation='Sigmoid')
        
        
    def forward(self,input_dict:Dict[str,Tensor]) \
            -> Dict[str,Tensor]:
        output_dict = {}
        output_dict["raw_rgbs"] = self.color_linear(super().forward(input_dict)) 
        return output_dict

@gin.configurable
class ColorField(BaseField):
    def __init__(self,
                 encoder_cls:Encoder,
                 mlp_cls:VanillaMLP,
                 input_components: Dict[str,int]
                 ):
        super(ColorField,self).__init__(encoder_cls,mlp_cls,input_components)

    def forward(self,input_dict:Dict[str,Tensor]) \
            -> Dict[str,Tensor]:
        output_dict = {}
        output_dict["raw_rgbs"] = super().forward(input_dict) 
        return output_dict


if __name__ == "__main__":
    encoder = PositionalEncoder(10)
    mlp = VanillaMLP(dims=[encoder.out_dim,10,10,10], skip_layers=[],activation=nn.Sigmoid(), out_activation=nn.Softplus(beta=100))
    input = torch.rand((3,3)) # [batch size, input size]
    geo_field = GeometricField(encoder, mlp,['enc_feature'])
    
    print(list(geo_field.parameters()))