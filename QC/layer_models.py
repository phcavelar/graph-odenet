import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from mpnn import MPNN_enn_edge as MPNN_enn
from set2set import Set2Set
from layers import EdgeEncoderMLP, EdgeGraphConvolution
from torch_scatter import scatter_add

class MPNN_ENN_K_Sum(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(MPNN_ENN_K_Sum, self).__init__()
        self.input = nn.Linear(in_features=node_features,out_features=hidden_features)
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        self.mpnn = MPNN_enn(edge_features, hidden_features)
        self.mpnn.set_T(num_layers)
        self.output = nn.Linear(in_features=hidden_features,out_features=out_features)
        self.type = type

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
        if self.type=="classification":
            x = F.log_softmax(x,dim=1)
        return x

class MPNN_ENN_K_Set2Set(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(MPNN_ENN_K_Set2Set, self).__init__()
        self.input = nn.Linear(in_features=node_features,out_features=hidden_features)
        self.ee = EdgeEncoderMLP( edge_features, hidden_features )
        self.mpnn = MPNN_enn(edge_features, hidden_features)
        self.mpnn.set_T(num_layers)
        self.s2s = Set2Set(hidden_features, s2s_processing_steps, num_layers=1)
        self.output = nn.Linear(in_features=hidden_features,out_features=out_features)
        self.type = type

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
        if self.type=="classification":
            x = F.log_softmax(x,dim=1)
        return x


class EdgeGCN_K_Sum(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(EdgeGCN_K_Sum, self).__init__()
        if num_layers<2:
            raise ValueError("Num layers must be at least 2")
        self.gcin = EdgeGraphConvolution( node_features, hidden_features )
        self.gcmid = nn.ModuleList(
                [ EdgeGraphConvolution( hidden_features, hidden_features )
                        for _ in range(num_layers-2) ] )
        self.gcout = EdgeGraphConvolution( hidden_features, out_features )
        self.dropout = dropout
        
        self.eein = EdgeEncoderMLP( edge_features, hidden_features )
        self.eemid = EdgeEncoderMLP( edge_features, hidden_features ) if num_layers >=3 else None
        self.eeout = EdgeEncoderMLP( edge_features, out_features )
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
        efin = self.eein(edge_features) 
        if self.eemid:
            efmid = self.eemid(edge_features)
        #end if
        efout = self.eeout(edge_features) 

        x = F.relu(self.gcin(x, Esrc, Etgt, efin))
        x = F.dropout(x, self.dropout, training=self.training)
        
        for gc in self.gcmid:
            x = F.relu(gc(x, Esrc, Etgt, efmid))
            x = F.dropout(x, self.dropout, training=self.training)
        #end for
        
        x = self.gc3(x, Esrc, Etgt, efout)
        x = scatter_add(x, batch, dim=0, dim_size=batch_size)
        return F.log_softmax(x, dim=1) if self.type == "classification" else x


class EdgeGCN_K_Set2Set(nn.Module):
    def __init__(self, node_features = None, edge_features = None, target_features = 1, hidden_features = 73, num_layers = 3, s2s_processing_steps = 12, type="regression", dropout=0.5, **kwargs):
        super(EdgeGCN_K_Set2Set, self).__init__()
        if num_layers<2:
            raise ValueError("Num layers must be at least 2")
        self.gcin = EdgeGraphConvolution( node_features, hidden_features )
        self.gcmid = nn.ModuleList(
                [ EdgeGraphConvolution( hidden_features, hidden_features )
                        for _ in range(num_layers-2) ] )
        self.gcout = EdgeGraphConvolution( hidden_features, out_features )
        self.dropout = dropout
        
        self.eein = EdgeEncoderMLP( edge_features, hidden_features )
        self.eemid = EdgeEncoderMLP( edge_features, hidden_features ) if num_layers >=3 else None
        self.eeout = EdgeEncoderMLP( edge_features, out_features )
        
        self.s2s = Set2Set(out_features, s2s_processing_steps, num_layers=1)
    #end __init__
    
    def forward(self,
            node_features,     # N x fn :: Float
            edge_features,     # E x fe :: Float
            Esrc,    # E :: Long
            Etgt,    # N x E :: Float
            batch,   # B x N :: Float
            ):
        x = node_features
        efin = self.eein(edge_features) 
        if self.eemid:
            efmid = self.eemid(edge_features)
        #end if
        efout = self.eeout(edge_features) 

        x = F.relu(self.gcin(x, Esrc, Etgt, efin))
        x = F.dropout(x, self.dropout, training=self.training)
        
        for gc in self.gcmid:
            x = F.relu(gc(x, Esrc, Etgt, efmid))
            x = F.dropout(x, self.dropout, training=self.training)
        #end for
        
        x = self.gc3(x, Esrc, Etgt, efout)
        x = self.s2s(x, batch)[:,:x.size()[1]]
        return F.log_softmax(x, dim=1) if self.type == "classification" else x
