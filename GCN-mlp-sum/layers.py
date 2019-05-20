import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MyLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is not None:
            return torch.mm(input, self.weight) + self.bias
        else:
            return torch.mm(input, self.weight)


class NonLinear(Module):
    def __init__(self, in_features, out_features, bias=True, f=F.relu):
        super(NonLinear, self).__init__()
        self.linear = MyLinear(in_features,out_features,bias=bias)
        self.bias = bias
        self.f=f
    #end __init__
   
    def forward(self, input):
        return self.f(self.linear(input))
    #end forward
#end NonLinear

class MLP(Module):
    def __init__(self, in_features, layer_sizes, out_features = None, bias=True):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = layer_sizes[-1]
            layer_sizes = layer_size[:-1]
        layer_inputs = [in_features] + layer_sizes[:-1]
        layers_ = [
                NonLinear(in_d, out_d, bias=bias)
                  for in_d, out_d in zip(layer_inputs,layer_sizes)
        ] + [ MyLinear( layer_sizes[-1], out_features, bias=bias ) ]
        self.layers = nn.Sequential( *layers_ )
    #end __init__
    
    def forward(self, input):
        return self.layers(input)
    #end forward
#end MLP

class GraphConvolution(Module):
    """
    MLP GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = MLP( in_features, [out_features], out_features, bias=bias )

    def forward(self, input, adj):
        support = self.mlp(input)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               


class FixedGraphConvolution(Module):
    """
    MLP GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(FixedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = MLP( in_features, [out_features], out_features, bias=bias )
        self.adj = torch.Tensor( [[1]] )

    def forward(self, input):
        support = self.mlp(input)
        output = torch.spmm(self.adj, support)
        return output
            
    def set_adj(self,adj):
        self.adj = adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
