import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
import gin
from typing import List, Callable
import tinycudann as tcnn
from utils.logger import *

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))
trunc_exp = _trunc_exp.apply

def linear_layer(in_features:int,out_features:int,first_layer:bool, last_layer:bool,
                 geometric_init:bool,weight_norm:bool, use_freq:bool, bias:float=1.):
    linear = nn.Linear(in_features,out_features)
    if geometric_init:
        if last_layer:
            torch.nn.init.normal_(linear.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_features), std=0.0001)
            torch.nn.init.constant_(linear.bias, bias)
        elif first_layer and use_freq:
            torch.nn.init.constant_(linear.bias, 0.0)
            torch.nn.init.constant_(linear.weight[:, 3:], 0.0)
            torch.nn.init.normal_(linear.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_features))
        else:
            torch.nn.init.constant_(linear.bias, 0.0)
            torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(out_features))
    if weight_norm:
        linear = nn.utils.weight_norm(linear)
    return linear    

def activation_fn(activation_str:str):
    """
    relu, sigmoid, softplus
    """
    if activation_str == 'relu' or activation_str == 'ReLU':
        return nn.ReLU()
    elif activation_str == 'sigmoid' or activation_str == 'Sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'softplus' or activation_str == 'Softplus':
        return nn.Softplus(beta=100)
    elif activation_str == 'tanh' or activation_str == 'Tanh':
        return nn.Tanh()
    elif activation_str == None or activation_str == 'None':
        return None
    elif activation_str == 'trunc_exp':
        return trunc_exp
    else:
        LOG_ERROR("Invalid activation name: %s" % activation_str)
        return None

@gin.configurable
class VanillaMLP(nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 num_hidden_layers:int,
                 hidden_dim:int,
                 output_dim:int, 
                 activation:str,
                 out_activation:str=None,
                 skip_layers:List[int]=[], 
                 geometric_init:bool=True,
                 weight_norm:bool=False,
                 use_freq:bool=True,
                 bias:float=1.
                 ):
        super(VanillaMLP, self).__init__()
        """
        Args:
            dims: list[int], MLP's dimensions = [input_dim, hidden_dim1,hidden_dim2,..., output_dim]
            skip_layers: list[int], input of skip Layers = concat([prev output, first input])
            activation: str, activation function such as ReLU, Sigmoid, Softplus,...
            out_activation: str, activation function of last layer such as ReLU, Sigmoid, Softplus,...
            geometric_init: bool, geometric_initializaion
            weight_norm: bool, weight normalization
            use_freq: bool, use frequency encoding(positional encoding)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers 
        self.skip_layers = skip_layers
        layers = []
        # input layer
        if num_hidden_layers > 0:
            layers.append(linear_layer(input_dim,hidden_dim,True,False,geometric_init,weight_norm,use_freq,bias))
        else:
            layers.append(linear_layer(input_dim,output_dim,True,False,geometric_init,weight_norm,use_freq,bias))
        ## hidden layer
        for i in range(0, num_hidden_layers-1):
            # initialize parameters
            if i + 1 in skip_layers:
                layers.append(linear_layer(hidden_dim + input_dim, hidden_dim,False,False,\
                geometric_init,weight_norm,use_freq,bias))
            else:
                layers.append(linear_layer(hidden_dim,hidden_dim,False,False,\
                geometric_init,weight_norm,use_freq,bias))
        # output layer
        if num_hidden_layers > 0:
            layers.append(linear_layer(hidden_dim,output_dim,False,False,\
                geometric_init,weight_norm,use_freq,bias))
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(self.layers)
        self.activation = activation_fn(activation)
        self.out_activation = activation_fn(out_activation)

    def forward(self, input:Tensor):
        x = input
        for i,layer in enumerate(self.layers):
            if i in self.skip_layers:
                x = torch.cat([x,input],1)
            x = layer(x)
            if i < len(self.layers)-1:
                x = self.activation(x)
        if self.out_activation:
            x = self.out_activation(x)
        return x

@gin.configurable
class TcnnMLP(nn.Module):
    def __init__(self,
            input_dim:int, 
            num_hidden_layers:int,
            hidden_dim:int,
            output_dim:int, 
            activation:str,
            out_activation:str=None):
        super(TcnnMLP,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None" if out_activation is None else out_activation,
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_hidden_layers,
            })
    
    def forward(self, x:Tensor) -> Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    mlp = VanillaMLP(dims=[3,10,10,2], skip_layers=[],activation='relu', out_activation='softplus')
    input = torch.rand((3,3)) # [batch size, input size]
    print(input)
    output = mlp(input)
    print(output.shape)
    print(list(mlp.parameters()))


