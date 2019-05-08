import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class MPNN_enn(nn.Module):
    def __init__(self, edge_net_hidden_dim, edge_net_hidden_dim, node_data_hidden_dim=200, readout_hidden_dim, output_dim):
        raise NotImplementedError("Not done yet.")
        super(MPNN_enn, self).__init__()
        
        self.e_d = edge_data_dim
        self.en_d = edge_net_hidden_dim
        self.h_d = node_data_hidden_dim
        self.r_d = readout_hidden_dim
        self.o_d = output_dim

        self.edge_net = nn.Sequential(
                nn.Linear(self.e_d,self.en_d),
                nn.Relu(),
                nn.Linear(self.en_d,self.h_d*self.h_d),
                nn.Relu(),
        )
        
        self.update_net = nn.GRUCell( self.h_d*2, self.h_d )

    def forward(self, x, adj, T=8, edge_data=None):
        if edge_data is None:
            raise ValueError( "Nede to pass edge_data for every edge" )
        edge_A = {}
        for v in range(adj.shape[0]):
            for w in range(adj.shape[1]):
                if adj[v,w]:
                    edge_A[v,w] = self.edge_net(edge_data[v,w]).reshape(self.h_d, self.h_d)
                #end if
            #end for
        #end for
        GRU_h = torch.zeros( x.shape )
        for t in range(T):
            m = torch.zeros( x.shape, device=self.device )
            for v in range(adj.shape[0]):
                for w in range(adj.shape[1]):
                    if adj[v,w]:
                        m[v] += self.message(x,v,w,edge_data=edge_A)
                    #end if
                #end for
            #end for
            x = self.update_net(torch.cat([x, m], 1), GRU_h)
        #end for
        
        return F.log_softmax(x, dim=1)
        
    def message(self,x,tgt,src,edge_data=None):
        return torch.mm( edge_data[tgt,src], x[src] )
