import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

class MLP(nn.Module):
    def __init__(self, input, layers, output):
        super(MLP, self).__init__()
        ds = [input] + layers + [output]
        self.hidden_linear = [nn.Linear(d_i,d_o,bias=True) for d_i, d_o in zip(ds[:-2],ds[1:-1])]
        self.output_linear = nn.Linear(ds[-2],ds[-1],bias=True)
        seq = []
        for l in self.hidden_linear:
            seq.append(l)
            seq.append(nn.ReLU())
        seq.append(self.output_linear)
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)
#end MLP

class IN(nn.Module):
    """
      Interaction network according to Battaglia et al's paper "Interaction Networks for Learning about Objects,Relations and Physics"
      n is the number of objects
      m is the number of relations
      Msrc is the matrix indicating that object i is the source of relation r, of shape (n,m)
      Mtgt is the matrix indicating that object i is the source of relation r, of shape (n,m)
      O is the Object input matrix of shape (n,d_O)
      R is the Relation input matrix of shape (m,d_R)
      X is the external effect input matrix of shape (n,d_X)
      
      There are two learned functions fR and fO.
      fR: (d_R+2*d_O) -> d_E
      fO: (d_E+d_X) -> d_P
      
      Where
      E = fR(R'), the "effects" matrix, is of shape (m,d_E)
      P = fO(E'), the "prediction" matrix, is of shape (n,d_P)
      R' is the concatenation of R with Msrc.T x O and Mtgt.T x O, and is of shape (m,d_R+2*d_O)
      E' is the concatenation of Mtgt x E with X
    """
    def __init__(self, d_O, d_R, d_X, d_P):
        super(IN, self).__init__()
        
        E_num_hidden_layers = 4
        d_E_hidden = 150
        d_E = 50
        
        P_num_hidden_layers = 1
        d_P_hidden = 100
        
        self.fR = MLP(d_R+2*d_O, E_num_hidden_layers*[d_E_hidden], d_E)
        self.fO = MLP(d_E+d_X, P_num_hidden_layers*[d_P_hidden], d_P)

    def forward(self, O, R, X, Msrc, Mtgt):
        Rsrc = torch.matmul(Msrc.t(), O)
        Rtgt = torch.matmul(Mtgt.t(), O)
        R_prime = torch.cat( ([R] if R is not None else [])+[Rsrc,Rtgt], dim=1 )
        E = self.fR(R_prime)
        E_prime = torch.cat( [torch.matmul(Mtgt,E)]+([X] if X is not None else []), dim=1)
        P = self.fO(E_prime)
        return P
#end IN


class IN_ODEfunc(nn.Module):
    def __init__(self, d_O, d_R, d_X, d_P):
        super(IN_ODEfunc, self).__init__()
        E_num_hidden_layers = 4
        d_E_hidden = 150
        d_E = 50
        
        P_num_hidden_layers = 1
        d_P_hidden = 100
        
        # TODO: Allow relation and external objects
        d_X = 0
        d_R = 0
        
        self.fR = MLP(d_R+2*d_O+1, E_num_hidden_layers*[d_E_hidden], d_E)
        self.fO = MLP(d_E+d_X, P_num_hidden_layers*[d_P_hidden], d_P)
        self.nfe = 0
        
    def set_fixed(self,Otail,Msrc,Mtgt):
        # TODO: Allow relation and external objects
        self.Ofixed = Otail
        self.Msrc = Msrc
        self.Mtgt = Mtgt

    def forward(self, t, x):
        self.nfe += 1
        # Concatenate time to node's features
        O = torch.cat([x,self.Ofixed], dim=1)
        Rsrc = torch.matmul(self.Msrc.t(), O)
        Rtgt = torch.matmul(self.Mtgt.t(), O)
        tt = torch.ones_like(Rsrc[:, :1]) * t
        R_prime = torch.cat([Rsrc,Rtgt,tt], dim=1)
        E = self.fR(R_prime)
        E_prime = torch.matmul(self.Mtgt,E)
        P = self.fO(E_prime)
        return P

class IN_ODE(nn.Module):
    """
      Interaction network according to Battaglia et al's paper "Interaction Networks for Learning about Objects,Relations and Physics"
      n is the number of objects
      m is the number of relations
      Msrc is the matrix indicating that object i is the source of relation r, of shape (n,m)
      Mtgt is the matrix indicating that object i is the source of relation r, of shape (n,m)
      O is the Object input matrix of shape (n,d_O)
      R is the Relation input matrix of shape (m,d_R)
      X is the external effect input matrix of shape (n,d_X)
      
      There are two learned functions fR and fO.
      fR: (d_R+2*d_O) -> d_E
      fO: (d_E+d_X) -> d_P
      
      Where
      E = fR(R'), the "effects" matrix, is of shape (m,d_E)
      P = fO(E'), the "prediction" matrix, is of shape (n,d_P)
      R' is the concatenation of R with Msrc.T x O and Mtgt.T x O, and is of shape (m,d_R+2*d_O)
      E' is the concatenation of Mtgt x E with X
    """
    def __init__(self, d_O, d_R, d_X, d_P,tol=1e-5):
        super(IN_ODE, self).__init__()
        
        E_num_hidden_layers = 4
        d_E_hidden = 150
        d_E = 50
        
        P_num_hidden_layers = 1
        d_P_hidden = 100
        
        self.fR = MLP(d_R+2*d_O, E_num_hidden_layers*[d_E_hidden], d_E)
        self.fO = MLP(d_E+d_X, P_num_hidden_layers*[d_P_hidden], d_P)
        self.odefunc = IN_ODEfunc(d_O, d_R, d_X, d_P)
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol
        
        self.d_P = d_P

    def forward(self, O, R, X, Msrc, Mtgt,tol=None):
        Otail = O[:,self.d_P:]
        Ohead = O[:,:self.d_P]
        self.integration_time = self.integration_time.type_as(O)
        self.odefunc.set_fixed(Otail,Msrc,Mtgt)
        P = odeint(self.odefunc, Ohead, self.integration_time, rtol=self.tol, atol=self.tol)
        return P
        
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
#end IN
