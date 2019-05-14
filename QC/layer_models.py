import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from mpnn import MPNN_enn_edge as MPNN_enn
from set2set import Set2Set
from layers import TransitionMLP, EdgeEncoderMLP, EdgeGraphConvolution
from torch_scatter import scatter_add

def get_output_function(type,target_features):
      if type=="regression":
          return lambda x: x
      elif target_features == 1:
          return lambda x: x
      else:
          return lambda x: F.log_softmax(x,dim=1)
#end get_output_function

class UnimplementedModel(nn.Module):
    def __init__(self,*args,**kwargs):
        raise NotImplementedError("Model not implemented yet")
    def forward(self,*args,**kwargs):
        raise NotImplementedError("Model not implemented yet")
#end UnimplementedModel
    

class MPNN_ENN_K_Sum(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(MPNN_ENN_K_Sum, self).__init__()
        self.input = nn.Linear(in_features=node_features,out_features=hidden_features)
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        self.mpnn = MPNN_enn(edge_features, hidden_features)
        self.mpnn.set_T(num_layers)
        self.output = nn.Linear(in_features=hidden_features,out_features=target_features)
        self.type = type
        self.output_function = get_output_function(type,target_features)

    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        batch_size = batch.max().item() + 1
        x = node_features
        edge_data = self.ee(edge_features)
        
        x = self.input(x)
        x = self.mpnn(x,Esrc,Etgt,edge_data)
        x = self.output(x)
        x = scatter_add(x, batch, dim=0, dim_size=batch_size)
        return self.output_function(x)

class MPNN_ENN_K_Set2Set(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(MPNN_ENN_K_Set2Set, self).__init__()
        self.input = nn.Linear(in_features=node_features,out_features=hidden_features)
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        self.mpnn = MPNN_enn(edge_features, hidden_features)
        self.mpnn.set_T(num_layers)
        self.s2s = Set2Set(hidden_features, s2s_processing_steps, num_layers=1)
        self.output = nn.Linear(in_features=hidden_features,out_features=target_features)
        self.type = type
        self.output_function = get_output_function(type,target_features)

    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        x = node_features
        edge_data = self.ee(edge_features)
        
        x = self.input(x)
        x = self.mpnn(x,Esrc,Etgt,edge_data)
        x = self.s2s(x,batch)[:,:x.size()[1]]
        x = self.output(x)
        return self.output_function(x)


class EdgeGCN_K_Sum(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(EdgeGCN_K_Sum, self).__init__()
        self.mlpin = TransitionMLP( node_features, hidden_features )
        self.gcmid = nn.ModuleList(
                [ EdgeGraphConvolution( hidden_features, hidden_features )
                        for _ in range(num_layers) ] )
        self.mlpout = TransitionMLP( hidden_features, target_features )
        self.dropout = dropout
        
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        
        self.type = type
        self.output_function = get_output_function(type,target_features)
    #end __init__
    
    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        batch_size = batch.max().item() + 1
        x = node_features
        ef = self.ee(edge_features)

        x = self.mlpin(x)
        
        for gc in self.gcmid[:-1]:
            x = F.relu(gc(x, Esrc, Etgt, ef))
            x = F.dropout(x, self.dropout, training=self.training)
        #end for
        x = self.gcmid[-1](x, Esrc, Etgt, ef)
        
        x = self.mlpout(x)
        
        x = scatter_add(x, batch, dim=0, dim_size=batch_size)
        return self.output_function(x)


class EdgeGCN_K_Set2Set(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(EdgeGCN_K_Set2Set, self).__init__()
        self.mlpin = TransitionMLP( node_features, hidden_features )
        self.gcmid = nn.ModuleList(
                [ EdgeGraphConvolution( hidden_features, hidden_features )
                        for _ in range(num_layers) ] )
        self.mlpout = TransitionMLP( hidden_features, target_features )
        self.dropout = dropout
        
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        
        self.s2s = Set2Set(hidden_features, s2s_processing_steps, num_layers=1)
        
        self.type = type
        self.output_function = get_output_function(type,target_features)
    #end __init__
    
    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        x = node_features
        ef = self.ee(edge_features)

        x = self.mlpin(x)
        
        for gc in self.gcmid[:-1]:
            x = F.relu(gc(x, Esrc, Etgt, ef))
            x = F.dropout(x, self.dropout, training=self.training)
        #end for
        x = self.gcmid[-1](x, Esrc, Etgt, ef)
        
        x = self.s2s(x, batch)[:,:x.size()[1]]
        x = self.mlpout(x)
        return self.output_function(x)
        

class EdgeRES1_K_Set2Set(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(EdgeRES1_K_Set2Set, self).__init__()
        self.mlpin = TransitionMLP( node_features, hidden_features )
        self.gcmid = RESKnorm( hidden_features, hidden_features, hidden_features, nlayers = num_layers, residue_layers=1 )
        self.mlpout = TransitionMLP( hidden_features, target_features )
        
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        
        self.s2s = Set2Set(hidden_features, s2s_processing_steps, num_layers=1)
        
        self.type = type
        self.output_function = get_output_function(type,target_features)
    #end __init__
    
    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        x = node_features
        ef = self.ee(edge_features)

        x = self.mlpin(x)
        
        x = self.gcmid(x, Esrc, Etgt, ef)
        
        x = self.s2s(x, batch)[:,:x.size()[1]]
        x = self.mlpout(x)
        return self.output_function(x)
        
class RESKnorm(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=3, residue_layers=1):
        super(RESKnorm, self).__init__()
        
        if nlayers<2+residue_layers:
            raise ValueError("Can't make a Residual GCN with less than {} layers using {} layers for each residual block".format(2+residue_layers,residue_layers))
        
        self.n_layers = nlayers
        stacked_layers = (
            [EdgeGraphConvolution(nfeat, nhid)] +
            [EdgeGraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2) ] +
            [EdgeGraphConvolution(nhid, nclass)]
        )
        self.gcs = nn.ModuleList(stacked_layers)
        self.norms = nn.ModuleList([nn.GroupNorm(min(32, nhid), nhid) for _ in range(self.n_layers-2)])
        self.residue_layers = residue_layers

    def forward(self, x, Esrc, Etgt, ef):
        gather_residue = 1
        
        for gc,norm in zip(self.gcs[0:-1],self.norms):
            gather_residue -= 1
            if gather_residue == 0:
                r = x
                gather_residue = self.residue_layers
            x = F.relu(gc(x, Esrc, Etgt, ef))
            x = norm(x)
            if gather_residue == 1:
                x = x + r
        #end for
        if gather_residue > 1:
            x = x + r
        x = self.gcs[-1](x, Esrc, Etgt, ef)
        return x

