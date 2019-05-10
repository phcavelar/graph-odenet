import torch
import torch.nn as nn
import torch.nn.functional as F

class MPNN_enn_edge(nn.Module):
    def __init__(self, edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim=200):
        super(MPNN_enn_edge, self).__init__()
        
        self.e_d = edge_data_dim
        self.en_d = edge_net_hidden_dim
        self.h_d = node_data_hidden_dim

        self.edge_net = nn.Sequential(
                nn.Linear(self.e_d,self.en_d),
                nn.ReLU(),
                nn.Linear(self.en_d,self.h_d*self.h_d),
                nn.ReLU(),
        )
        
        self.update_net = nn.GRUCell( self.h_d*2, self.h_d )

    def forward(self, x, adj, T=8, edge_data=None, edges=None):
        if edge_data is None:
            raise ValueError( "Need to pass edge_data for every edge" )
        edge_A = torch.zeros( [edges.size()[0],self.h_d,self.h_d], dtype=x.dtype, device=x.device )
        for e_id in range(edges.shape[0]):
            edge_A[e_id] = self.edge_net(edge_data[edges[e_id][0],edges[e_id][1]]).reshape(self.h_d, self.h_d)
        #end for
        #GRU_h = torch.zeros_like( x, device=x.device )
        for t in range(T):
            print("t=",t)
            m = torch.zeros_like( x, device=x.device )
            for v in range(adj.shape[0]):
                for e_id in range(edges.shape[0]):
                    if edges[e_id,:].eq(v).any():
                        m[v] += self.message(x,edges[e_id,0],edges[e_id,1],edge_data=edge_A[e_id])
                    #end if
                #end for
            #end for
            x = self.update_net(torch.cat([x, m], 1), x)
        #end for
        
        return x
        
    def message(self,x,src,tgt,edge_data=None):
        return torch.mm( edge_data, x[src].unsqueeze(1) ).squeeze()

class MPNN_enn(nn.Module):
    def __init__(self, edge_data_dim, edge_net_hidden_dim, node_data_hidden_dim=200):
        super(MPNN_enn, self).__init__()
        
        self.e_d = edge_data_dim
        self.en_d = edge_net_hidden_dim
        self.h_d = node_data_hidden_dim

        self.edge_net = nn.Sequential(
                nn.Linear(self.e_d,self.en_d),
                nn.ReLU(),
                nn.Linear(self.en_d,self.h_d*self.h_d),
                nn.ReLU(),
        )
        
        self.update_net = nn.GRUCell( self.h_d*2, self.h_d )

    def forward(self, x, adj, T=8, edge_data=None, edges=None):
        if edge_data is None:
            raise ValueError( "Need to pass edge_data for every edge" )
        edge_A = torch.zeros( [adj.size()[0],adj.size()[1],self.h_d,self.h_d], dtype=x.dtype, device=x.device )
        #edge_A = {}
        for v in range(adj.shape[0]):
            for w in range(adj.shape[1]):
                if adj[v,w]:
                    edge_A[v,w] = self.edge_net(edge_data[v,w]).reshape(self.h_d, self.h_d)
                #end if
            #end for
        #end for
        #GRU_h = torch.zeros_like( x, device=x.device )
        for t in range(T):
            print("t=",t)
            m = torch.zeros_like( x, device=x.device )
            for v in range(adj.shape[0]):
                for e_id in range(edges.shape[0]):
                    if edges[e_id,:].eq(v).any():
                        m[v] += self.message(x,edges[e_id,0],edges[e_id,1],edge_data=edge_A)
                    #end if
                #end for
            #end for
            x = self.update_net(torch.cat([x, m], 1), x)
        #end for
        
        return x
        
    def message(self,x,src,tgt,edge_data=None,directed=False):
        if directed:
            return torch.mm( edge_data[tgt,src], x[src].unsqueeze(1) ).squeeze()
        else:
            return torch.mm( edge_data[min(tgt,src),max(tgt,src)], x[src].unsqueeze(1) ).squeeze()
