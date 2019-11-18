import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn


class GraphConvolution(Module):
    """
    GAT layer
    """

    def __init__(self,in_features, out_features, bias=True,act=F.relu,eps=1e-6):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.f = nn.Linear(2*in_features,out_features)
        self.w = nn.Linear(2*in_features,1)
        self.eps = eps
        self.act = act
        self.reset_parameters()
        
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,src,tgt,Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.spmm(Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.spmm(Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               


class FixedGraphConvolution(Module):
    """
    GAT layer
    """

    def __init__(self,in_features, out_features, bias=True,act=F.relu,eps=1e-6):
        super(FixedGraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.f = nn.Linear(2*in_features,out_features)
        self.w = nn.Linear(2*in_features,1)
        self.eps = eps
        self.act = act
        self.reset_parameters()
        self.src = torch.Tensor( [[1]] )
        self.tgt = torch.Tensor( [[1]] )
        self.Mtgt = torch.Tensor( [[1]] )
        
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
            
    def set_adj(self,src,tgt,Mtgt):
        self.src = src
        self.tgt = tgt
        self.Mtgt = Mtgt
    
    def forward(self,x):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[self.src] # E,i
        htgt = x[self.tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = self.act(self.f(h)) # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.spmm(self.Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.spmm(self.Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
