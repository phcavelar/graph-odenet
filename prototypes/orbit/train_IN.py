import os
import tqdm

from pprint import pprint as pp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from prepare_dataset import get_epoch

from model import IN, IN_ODE

TIMESTEP_TYPES = ["s", "e"]  # start and end
VAR_NAMES = ["p", "v", "m", "r", "f", "d"]
VAR_FILENAMES = ["pos", "vel", "mass", "radii", "force", "data"]


def read_instance(data_folder, simulation, timestep):
    i = {}
    for ttype, tstep in zip(TIMESTEP_TYPES, [timestep, timestep+1]):
        for var, fname in zip(VAR_NAMES, VAR_FILENAMES):
            i[ttype+"_"+var] = np.load("{folder}/{sim}/{tstep}.{var}.npy".format(
                folder=data_folder,
                sim=simulation,
                tstep=tstep,
                var=fname
            )
            )
        # end for
    # end for

    return i
# end read_instance


def get_O(instance, ttype):
    return np.concatenate(
        [instance[ttype+"_v"], instance[ttype+"_p"], instance[ttype+"_m"]],
        axis=1
    )
# end get_O


def process_instance(instance):
    Oin = get_O(instance, "s")
    Oout = get_O(instance, "e")
    float_dtype = Oout.dtype

    n = Oin.shape[0]
    Adj_matrix = np.ones([n, n]) - np.eye(n)
    relations = [(src, tgt) for src in range(n)
                 for tgt in range(n) if Adj_matrix[src, tgt] != 0]
    m = len(relations)

    Msrc = np.zeros([n, m], dtype=float_dtype)
    Mtgt = np.zeros([n, m], dtype=float_dtype)

    for r, st in enumerate(relations):
        s, t = st
        Msrc[s, r] = 1
        Mtgt[t, r] = 1

    return Oin, Oout, Msrc, Mtgt
# end process_instance


# TODO Make training schedule as in the paper
# TODO Train and test for the 10 folds
if __name__ == "__main__":
    DATASET_FOLDER = "./dataset/6"
    NUM_FOLDS = 10
    NUM_EPOCHS = 2000
    BATCH_SIZE = 100
    TRAIN_SIZE = 999000
    O_SHAPE = 5
    LEARNING_RATE = 0.001
    LR_DECAY = 0.8
    LR_DECAY_WINDOW = 40
    L2_NORM = 5e-4 # Not provided in paper, took from our other experiments
    PREDICTED_VALUES = 2 # vx, vy
    Model = IN

    for fold in tqdm.trange(NUM_FOLDS, desc="Fold"):    
        current_lr = LEARNING_RATE
        model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
        optimizer = optim.Adam(model.parameters(),
                           lr=current_lr, weight_decay=L2_NORM)
        for epoch in tqdm.trange(NUM_EPOCHS, desc="Epoch"):
            model.train()
            for b, batch in tqdm.tqdm(get_epoch("{}/{}/train".format(DATASET_FOLDER,fold),BATCH_SIZE), total=TRAIN_SIZE/BATCH_SIZE, desc="Batch Train"):
                bOin, bOout, bMsrc, bMtgt, n_list, m_list = batch
                Pred = model(bOin, None, None, bMsrc, bMtgt)
                optimizer.zero_grad()
                loss = F.mse_loss(Pred, bOout[:,:PREDICTED_VALUES])
                loss.backward()
                optimizer.step()
                tqdm.write(loss)
            #end for
            
            model.eval()
            val_loss = []
            for b, batch in tqdm.tqdm(get_epoch("{}/{}/validation".format(DATASET_FOLDER,fold),BATCH_SIZE), total=TRAIN_SIZE/BATCH_SIZE, desc="Batch Valid"):
                bOin, bOout, bMsrc, bMtgt, n_list, m_list = batch
                with torch.no_grad():
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    loss = F.mse_loss(Pred, bOout[:,:PREDICTED_VALUES])
                #end with
                val_loss.append(loss)
            #end for
            
            val_loss = np.mean(val_loss)
            tqdm.write(val_loss)
        #end for
    #end for
