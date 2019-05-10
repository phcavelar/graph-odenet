import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from mpnn import MPNN_enn_edge as MPNN_enn
from set2set import Set2Set
from layers import EdgeEncoderMLP, EdgeGraphConvolution
from torch_scatter import scatter_add

class MPNN_ENN_Sum(nn.Module):
    def __init__(self, node_data_dim, edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim, output_dim, processing_steps=12, type="regression"):
        super(MPNN_ENN_Sum, self).__init__()
        self.input = nn.Linear(in_features=node_data_dim,out_features=node_data_hidden_dim)
        self.mpnn = MPNN_enn(edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim)
        self.s2s = Set2Set(node_data_hidden_dim, processing_steps, num_layers=1)
        self.output = nn.Linear(in_features=2*node_data_hidden_dim,out_features=output_dim)
        self.type = type

    def forward(self, x, adj, T=8, edge_data=None, edges=None, batch=None):
        if edge_data is None:
            raise ValueError( "Need to pass edge_data for every edge" )
        if batch is None:
            raise ValueError( "Need to specify which node is from which batch for set2set" )
        print("x",x.size())
        x = self.input(x)
        print("in",x.size())
        x = self.mpnn(x,adj,T,edge_data,edges)
        print("mpnn",x.size())
        x = self.s2s(x,batch)
        print("s2s",x.size())
        x = self.output(x)
        print("out",x.size())
        if self.type=="classification":
            x = F.log_softmax(x,dim=1)
        return x

class MPNN_ENN_Set2Set(nn.Module):
    def __init__(self, node_data_dim, edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim, output_dim, processing_steps=12, type="regression"):
        super(MPNN_ENN_Seq2Seq, self).__init__()
        self.input = nn.Linear(in_features=node_data_dim,out_features=node_data_hidden_dim)
        self.mpnn = MPNN_enn(edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim)
        self.s2s = Set2Set(node_data_hidden_dim, processing_steps, num_layers=1)
        self.output = nn.Linear(in_features=2*node_data_hidden_dim,out_features=output_dim)
        self.type = type

    def forward(self, x, adj, T=8, edge_data=None, edges=None, batch=None):
        if edge_data is None:
            raise ValueError( "Need to pass edge_data for every edge" )
        if batch is None:
            raise ValueError( "Need to specify which node is from which batch for set2set" )
        print("x",x.size())
        x = self.input(x)
        print("in",x.size())
        x = self.mpnn(x,adj,T,edge_data,edges)
        print("mpnn",x.size())
        x = self.s2s(x,batch)
        print("s2s",x.size())
        x = self.output(x)
        print("out",x.size())
        if self.type=="classification":
            x = F.log_softmax(x,dim=1)
        return x



class EdgeGCN3_Sum(nn.Module):
    def __init__( self, node_features, edge_features, hidden_features, out_features, dropout=0, type="regression" ):
        super(EdgeGCN3_Sum, self).__init__()
        self.gc1 = EdgeGraphConvolution( node_features, hidden_features )
        self.gc2 = EdgeGraphConvolution( hidden_features, hidden_features )
        self.gc3 = EdgeGraphConvolution( hidden_features, out_features )
        self.dropout = dropout
        
        self.ee1 = EdgeEncoderMLP( edge_features, hidden_features )
        self.ee2 = EdgeEncoderMLP( edge_features, hidden_features )
        self.ee3 = EdgeEncoderMLP( edge_features, out_features )
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
        
        ef1 = self.ee1(edge_features)
        x = F.relu(self.gc1(x, Esrc, Etgt, ef1))
        x = F.dropout(x, self.dropout, training=self.training)
        
        ef2 = self.ee2(edge_features)
        x = F.relu(self.gc2(x, Esrc, Etgt, ef2))
        x = F.dropout(x, self.dropout, training=self.training)
        
        ef3 = self.ee3(edge_features)
        x = self.gc3(x, Esrc, Etgt, ef3)
        x = scatter_add(x, batch, dim=0, dim_size=batch_size)
        return F.log_softmax(x, dim=1) if self.type == "classification" else x


class EdgeGCN3_Set2Set(nn.Module):
    def __init__( self, node_features, edge_features, hidden_features, out_features, dropout=0, processing_steps=8, type="regression" ):
        super(EdgeGCN3_Set2Set, self).__init__()
        self.gc1 = EdgeGraphConvolution( node_features, hidden_features )
        self.gc2 = EdgeGraphConvolution( hidden_features, hidden_features )
        self.gc3 = EdgeGraphConvolution( hidden_features, out_features )
        self.s2s = Set2Set(out_features, processing_steps, num_layers=1)
        self.dropout = dropout
        
        self.ee1 = EdgeEncoderMLP( edge_features, hidden_features )
        self.ee2 = EdgeEncoderMLP( edge_features, hidden_features )
        self.ee3 = EdgeEncoderMLP( edge_features, out_features )
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
        
        ef1 = self.ee1(edge_features)
        x = F.relu(self.gc1(x, Esrc, Etgt, ef1))
        x = F.dropout(x, self.dropout, training=self.training)
        
        ef2 = self.ee2(edge_features)
        x = F.relu(self.gc2(x, Esrc, Etgt, ef2))
        x = F.dropout(x, self.dropout, training=self.training)
        
        ef3 = self.ee3(edge_features)
        x = self.gc3(x, Esrc, Etgt, ef3)
        x = self.s2s(x, batch)[:,:x.size()[1]]
        return F.log_softmax(x, dim=1) if self.type == "classification" else x
