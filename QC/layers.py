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

class EdgeEncoderMLP(Module):
    def __init__(self, edge_features, node_features, bias=True):
        super(EdgeEncoderMLP, self).__init__()
        self.mlp = MLP(  edge_features, [(edge_features+node_features*node_features)//2], node_features*node_features )
        self.nf = node_features
    #end __init__
    
    def forward(self, input):
        return self.mlp(input).reshape([input.size()[0],self.nf,self.nf])
    #end forward
#end MLP

class EdgeGraphConvolution_UNUSED(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_in_features, node_layers = 1, edge_layers = 1, bias=True):
        raise NotImplementedError("Not implemented")
        super(EdgeGraphConvolution_UNUSED, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_f = MLP( in_features, [out_features for _ in range( node_layers )] )
        self.edge_f = MLP( in_features, [out_features for _ in range( edge_layers-1 )] + [ out_features * out_features ] )

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class EdgeGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, node_layers = 1, edge_layers = 1, bias=True):
        super(EdgeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

    def forward(self,
            input,     # N x fi :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            edge_data, # E x fo x fo :: Float
            ):
        support = torch.mm(input, self.weight) # N x fo
        edge_support = torch.index_select( support, 0, Esrc ) # E x fo
        edge_msg = torch.bmm( edge_data, edge_support.unsqueeze(-1) ).squeeze() # E x fo
        output = torch.spmm( Etgt, edge_msg ) # N x fo
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               


class FixedGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(FixedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adj = torch.Tensor( [[1]] )

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        output = torch.spmm(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
    def set_adj(self,adj):
        self.adj = adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
