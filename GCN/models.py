import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, FixedGraphConvolution
from torchdiffeq import odeint_adjoint as odeint


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCN3norm(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN3norm, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.norm2 = nn.GroupNorm(min(32, nhid), nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.norm2(x)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class RGCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RGCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        r = x
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x + r
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class RGCN3norm(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RGCN3norm, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.norm2 = nn.GroupNorm(min(32, nhid), nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        r = x
        x = self.gc2(x, adj)
        x = self.norm2(x)
        x = x + r
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class RGCN3fullnorm(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(RGCN3fullnorm, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.norm1 = nn.GroupNorm(min(32, nhid), nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.norm2 = nn.GroupNorm(min(32, nhid), nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.norm1(x)
        r = x
        x = self.gc2(x, adj)
        x = self.norm2(x)
        x = x + r
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


class GCNK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=2):
        super(GCNK, self).__init__()
        
        if nlayers<2:
            raise ValueError("Can't make a GCN with less than 2 layers")
        #end if
        
        self.n_layers = nlayers
        stacked_layers = (
            [GraphConvolution(nfeat, nhid)] +
            [ GraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2) ] +
            [GraphConvolution(nhid, nclass)]
        )
        self.gcs = nn.ModuleList(stacked_layers)
        self.dropout = dropout

    def forward(self, x, adj):
        for gc in self.gcs:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        #end for
        return F.log_softmax(x, dim=1)


class RESK1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=3):
        super(RESK1, self).__init__()
        
        if nlayers<3:
            raise ValueError("Can't make a Residual GCN with less than 3 layers using 1 layer for each residual block")
        #end if
        
        self.n_layers = nlayers
        stacked_layers = (
            [GraphConvolution(nfeat, nhid)] +
            [ GraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2) ] +
            [GraphConvolution(nhid, nclass)]
        )
        self.gcs = nn.ModuleList(stacked_layers)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcs[0](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for gc in self.gcs[1:-1]:
            r = x
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + r
        #end for
        x = self.gcs[-1](x, adj)
        return F.log_softmax(x, dim=1)


class RESK2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=4):
        super(RESK2, self).__init__()
        
        if nlayers<4:
            raise ValueError("Can't make a Residual GCN with less than 4 layers using 2 layers for each residual block")
        
        self.n_layers = nlayers
        stacked_layers = (
            [GraphConvolution(nfeat, nhid)] +
            [ GraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2) ] +
            [GraphConvolution(nhid, nclass)]
        )
        self.gcs = nn.ModuleList(stacked_layers)
        self.dropout = dropout
        self.residue_layers = 2

    def forward(self, x, adj):
        x = F.relu(self.gcs[0](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        gather_residue = 1
        
        for gc in self.gcs[1:-1]:
            gather_residue -= 1
            if gather_residue == 0:
                r = x
                gather_residue = self.residue_layers
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            if gather_residue == 1:
                x = x + r
        #end for
        if gather_residue > 1:
            x = x + r
        x = self.gcs[-1](x, adj)
        return F.log_softmax(x, dim=1)


class RESK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=3, residue_layers=1):
        super(RESK2, self).__init__()
        
        if nlayers<2+residue_layers:
            raise ValueError("Can't make a Residual GCN with less than {} layers using {} layers for each residual block".format(2+residue_layers,residue_layers))
        
        self.n_layers = nlayers
        stacked_layers = (
            [GraphConvolution(nfeat, nhid)] +
            [ GraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2) ] +
            [GraphConvolution(nhid, nclass)]
        )
        self.gcs = nn.ModuleList(stacked_layers)
        self.dropout = dropout
        self.residue_layers = residue_layers

    def forward(self, x, adj):
        x = F.relu(self.gcs[0](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        gather_residue = 1
        
        for gc in self.gcs[1:-1]:
            gather_residue -= 1
            if gather_residue == 0:
                r = x
                gather_residue = self.residue_layers
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            if gather_residue == 1:
                x = x + r
        #end for
        if gather_residue > 1:
            x = x + r
        x = self.gcs[-1](x, adj)
        return F.log_softmax(x, dim=1)


class ODEfunc(nn.Module):

    def __init__(self, dim, dropout):
        super(ODEfunc, self).__init__()
        self.norm1 =  nn.GroupNorm(min(32, dim), dim)
        self.gc1 = FixedGraphConvolution(dim+1, dim)
        self.dropout = dropout
        self.nfe = 0
        
    def set_adj(self,adj):
        self.gc1.set_adj( adj )
        #self.gc2.set_adj( adj )

    def forward(self, t, x):
        self.nfe += 1
        # Concatenate time to node's features
        x = self.norm1(x)
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        out = F.relu(self.gc1( ttx ))
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc, dropout, tol=1e-5):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol

    def forward(self, x, adj):
        self.integration_time = self.integration_time.type_as(x)
        self.odefunc.set_adj(adj)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        

class ODEGCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ODEGCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = ODEBlock(ODEfunc(nhid,dropout),dropout)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)
        

    @property
    def nfe(self):
        return self.gc2.nfe

    @nfe.setter
    def nfe(self, value):
        self.gc2.nfe = value
        

class ODEGCN3fullnorm(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ODEGCN3fullnorm, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.norm1 = nn.GroupNorm(min(32, nhid), nhid)
        self.gc2 = ODEBlock(ODEfunc(nhid,dropout),dropout)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.norm1(x)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)
        

    @property
    def nfe(self):
        return self.gc2.nfe

    @nfe.setter
    def nfe(self, value):
        self.gc2.nfe = value
