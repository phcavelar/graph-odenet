import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import IN, IN_ODE

TIMESTEP_TYPES = ["s","e"] #start and end
VAR_NAMES = ["p","v","m","r","f","d"]
VAR_FILENAMES = ["pos","vel","mass","radii","force","data"]

def read_instance(data_folder,simulation,timestep):
    i = {}
    for ttype,tstep in zip(TIMESTEP_TYPES,[timestep,timestep+1]):
        for var,fname in zip(VAR_NAMES,VAR_FILENAMES):
            i[ttype+"_"+var] = np.load( "{folder}/{sim}/{tstep}.{var}.npy".format(
                    folder = data_folder,
                    sim = simulation,
                    tstep = tstep,
                    var = fname
                    )
            )
        #end for
    #end for
    return i
#end read_instance

def get_O(instance,ttype):
    return np.concatenate(
        [instance[ttype+"_v"], instance[ttype+"_p"], instance[ttype+"_m"]],
        axis = 1
    )
#end get_O

def process_instance(instance):
    Oin = get_O(instance,"s")
    Oout = get_O(instance,"e")
    float_dtype = Oout.dtype
    n = Oin.shape[0]
    Adj_matrix = np.ones([n,n]) - np.eye(n)
    relations = [(src,tgt) for src in range(n) for tgt in range(n) if Adj_matrix[src,tgt]!=0]
    m = len(relations)
    Msrc = np.zeros([n,m],dtype=float_dtype)
    Mtgt = np.zeros([n,m],dtype=float_dtype)
    for r, st in enumerate(relations):
        s,t = st
        Msrc[s,r] = 1
        Mtgt[t,r] = 1
    return Oin, Oout, Msrc, Mtgt
#end process_instance
            

if __name__ == "__main__":
    DATA_FOLDER = "./data"
    MAX_TSTEP = 1000
    Model = IN
    
    simulations = sorted(os.listdir(DATA_FOLDER))
    inputs = []
    for sim in simulations:
        for t in range(MAX_TSTEP-1):
            inputs.append((sim,t))
        #end for
    #end for
    chosen = np.random.choice(len(inputs))
    bla = read_instance(DATA_FOLDER,*inputs[chosen])
    for k in bla:
        print("{}: {}".format(k,bla[k]))
    Oin, Oout, Msrc, Mtgt = process_instance(bla)
    Oin, Oout, Msrc, Mtgt = map(lambda x: x.astype(np.float32), [Oin, Oout, Msrc, Mtgt])
    for n,o in zip(["Oin","Oout","Msrc","Mtgt"],[Oin, Oout, Msrc, Mtgt]):
        print(n,o.shape,o[0])
    model = Model(Oin.shape[1],0,0,2)
    y = model(torch.tensor(Oin), None, None, torch.tensor(Msrc), torch.tensor(Mtgt))
    print(y)
