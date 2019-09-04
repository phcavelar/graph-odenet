import os
import shutil
import tqdm

import numpy as np

TIMESTEP_TYPES = ["s","e"] #start and end
VAR_NAMES = ["p","v","m","r","f","d"]
VAR_FILENAMES = ["pos","vel","mass","radii","force","data"]

def read_instance(data_folder,simulation):
    i = {}
    for var,fname in zip(VAR_NAMES,VAR_FILENAMES):
        i[var] = np.load( "{folder}/{sim}/{var}.npy".format(
                folder = data_folder,
                sim = simulation,
                var = fname
                )
        )
    #end for
    return i
#end read_instance

def get_O(instance,timestep):
    return np.concatenate(
        [instance["v"][timestep], instance["p"][timestep], instance["m"][timestep]],
        axis = 1
    )
#end get_O

def process_instance(instance, timestep):
    Oin = get_O(instance,timestep)
    Oout = get_O(instance,timestep+1)
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

# TODO Read processed instance from "dataset"
# TODO Make batch with block-diagonal matrix (How to make src and tgt matrix is in process_instance)
# TODO Make batch loader


if __name__ == "__main__":
    DATA_FOLDER = "./data"
    DATASET_FOLDER = "./dataset"
    MAX_TSTEP = 1000
    NUM_FOLDS = 10
    NUM_TRAIN_INSTANCES = .5#1000000
    NUM_VALIDATION_INSTANCES = .1#200000
    NUM_TEST_INSTANCES = .1#200000
    
    if os.path.isdir(DATASET_FOLDER):
        shutil.rmtree(DATASET_FOLDER)
    #end if
    
    print("Cleaning and preparing dataset folders")
    os.mkdir(DATASET_FOLDER)
    
    for fold in range(NUM_FOLDS):
        os.mkdir("{}/{}".format(DATASET_FOLDER,fold))
        for s in ["test","train","validation"]:
            os.mkdir("{}/{}/{}".format(DATASET_FOLDER,fold,s))
    #end
    
    simulations = [x for x in sorted(os.listdir(DATA_FOLDER)) if x != ".gitkeep"]
    values_size = len(simulations)*MAX_TSTEP
    Oin = get_O(read_instance(DATA_FOLDER,simulations[0]),0)
    input_shape = Oin.shape
    inputs = np.zeros([values_size,*input_shape])
    vidx = 0
    for sim in tqdm.tqdm(simulations):
        sim_instance = read_instance(DATA_FOLDER,sim)
        for t in tqdm.trange(MAX_TSTEP):
            Oin = get_O(sim_instance,t)
            inputs[vidx,...] = Oin[...]
            vidx+=1
        #end for
    #end for
    percentiles = [5,50,95]
    value_percentiles = np.zeros([len(percentiles),input_shape[-1]])
    
    for attridx in range(input_shape[-1]):
        value_percentiles[:,attridx] = np.percentile(inputs[:,:,attridx], percentiles)
    normalise = lambda x: ((2*((x-value_percentiles[1])/(value_percentiles[2]-value_percentiles[0])))-1)
    
    np.save(
            "{}/{}/normvals.np".format(DATASET_FOLDER,fold), value_percentiles
    )
    
    instances = [(sim,t) for sim in simulations for t in range(MAX_TSTEP-1)]
    n_train = int(len(instances)*NUM_TRAIN_INSTANCES)
    n_validation = int(len(instances)*NUM_VALIDATION_INSTANCES)
    n_test = int(len(instances)*NUM_TEST_INSTANCES)
    for fold in tqdm.tqdm(simulations, desc="Fold"):
        fold_instances = np.random.choice(len(instances), n_train + n_validation + n_test, replace=False)
        train = fold_instances[:n_train]
        validation = fold_instances[n_train:n_train + n_validation]
        test = fold_instances[n_train + n_validation:]
        for s, idxs in tqdm.tqdm(zip(["test","train","validation"],[test,train,validation]), desc="Split"):
            for i in tqdm.tqdm(idxs, desc=s):
                sim, tstep = instances[i]
                Oin, Oout, _0, _1 = process_instance(read_instance(DATA_FOLDER,sim),tstep)
                Omerged = np.append(Oin[np.newaxis,...],Oout[np.newaxis,...],axis=0)
                np.save(
                        "{}/{}/{}/{}.np".format(DATASET_FOLDER,fold,s, i), Omerged
                )
            #end
        #end
    #end
