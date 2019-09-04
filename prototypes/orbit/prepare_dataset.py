import os
import shutil
import tqdm
import fire

import numpy as np

TIMESTEP_TYPES = ["s", "e"]  # start and end
VAR_NAMES = ["p", "v", "m", "r", "f", "d"]
VAR_FILENAMES = ["pos", "vel", "mass", "radii", "force", "data"]


def read_instance(data_folder, simulation):
    i = {}
    for var, fname in zip(VAR_NAMES, VAR_FILENAMES):
        i[var] = np.load("{folder}/{sim}/{var}.npy".format(
            folder=data_folder,
            sim=simulation,
            var=fname
        ))
    # end for
    return i
# end read_instance


def get_O(instance, timestep):
    return np.concatenate(
        [instance["v"][timestep], instance["p"]
            [timestep], instance["m"][timestep]],
        axis=1
    )
# end get_O


def process_instance(instance, timestep):
    Oin = get_O(instance, timestep)
    Oout = get_O(instance, timestep+1)

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

def get_epoch(dataset_folder,batch_size=100):
    dataset_files = ["{}/{}".format(dataset_folder,x) for x in os.listdir(dataset_folder) if ".git" not in x]
    np.random.shuffle(dataset_files)
    for batch in gen_batch(dataset_files, batch_size=batch_size):
        yield batch
    #end for
#end get_epoch

def read_processed_instance(instance):
    Omerged = np.load(instance)
    return Omerged[0], Omerged[1]
#end read_processeed instance

def gen_batch(file_iterator, batch_size=100):
    batch=[]
    for f in file_iterator:
        batch.append(read_processed_instance(f))
        if len(batch)>=batch_size:
            yield merge_instances(batch)
            batch=[]
    yield merge_instances(batch)
#end gen_batch

def merge_instances(batch):
    batch_n = sum( (Oin.shape[0] for Oin, Oout in batch) )
    batch_m = sum( (Oin.shape[0]*Oin.shape[0]-Oin.shape[0] for Oin, Oout in batch) )
    float_dtype = batch[0][0].dtype
    O_shape = batch[0][0].shape[-1]
    bOin = np.zeros([batch_n,O_shape], float_dtype)
    bOout = np.zeros([batch_n,O_shape], float_dtype)
    bMsrc = np.zeros([batch_n,batch_m], float_dtype)
    bMtgt = np.zeros([batch_n,batch_m], float_dtype)
    n_list = []
    m_list = []
    n_acc = 0
    m_acc = 0
    for Oin, Oout in batch:
        n = Oin.shape[0]
        
        bOin[n_acc:n_acc+n,:] = Oin[:,:]
        bOout[n_acc:n_acc+n,:] = Oout[:,:]
        
        Adj_matrix = np.ones([n, n]) - np.eye(n)
        relations = [(src, tgt) for src in range(n)
                     for tgt in range(n) if Adj_matrix[src, tgt] != 0]

        m = len(relations)
        for r, (s, t) in enumerate(relations):
            bMsrc[n+s, m+r] = 1
            bMtgt[n+t, m+r] = 1
        #end for
        n_list.append(n)
        m_list.append(m)
    #end for
    return bOin, bOout, bMsrc, bMtgt, n_list, m_list
#end merge_instances

def prepare_dataset(num_of_bodies=6):
    DATA_FOLDER = "./data/{}".format(num_of_bodies)
    DATASET_FOLDER = "./dataset/{}".format(num_of_bodies)
    MAX_TSTEP = 1000
    NUM_FOLDS = 10
    NUM_TRAIN_INSTANCES = .5  # 1000000
    NUM_VALIDATION_INSTANCES = .1  # 200000
    NUM_TEST_INSTANCES = .1  # 200000

    if os.path.isdir(DATASET_FOLDER):
        shutil.rmtree(DATASET_FOLDER)

    print("Cleaning and preparing dataset folders")
    os.mkdir(DATASET_FOLDER)

    for fold in range(NUM_FOLDS):
        os.mkdir("{}/{}".format(DATASET_FOLDER, fold))
        for s in ["test", "train", "validation"]:
            os.mkdir("{}/{}/{}".format(DATASET_FOLDER, fold, s))
    # end

    simulations = [x for x in sorted(
        os.listdir(DATA_FOLDER)) if x != ".gitkeep"]
    values_size = len(simulations)*MAX_TSTEP
    Oin = get_O(read_instance(DATA_FOLDER, simulations[0]), 0)
    input_shape = Oin.shape
    inputs = np.zeros([values_size, *input_shape])
    vidx = 0
    for sim in tqdm.tqdm(simulations):
        sim_instance = read_instance(DATA_FOLDER, sim)
        for t in tqdm.trange(MAX_TSTEP):
            Oin = get_O(sim_instance, t)
            inputs[vidx, ...] = Oin[...]
            vidx += 1
        # end for
    # end for
    percentiles = [5, 50, 95]
    value_percentiles = np.zeros([len(percentiles), input_shape[-1]])

    for attridx in range(input_shape[-1]):
        value_percentiles[:, attridx] = np.percentile(
            inputs[:, :, attridx], percentiles)

    def normalise(x): return (
        (2*((x-value_percentiles[1])/(value_percentiles[2]-value_percentiles[0])))-1)

    np.save(
        "{}/{}/normvals.npy".format(DATASET_FOLDER, fold), value_percentiles
    )

    instances = [(sim, t) for sim in simulations for t in range(MAX_TSTEP-1)]
    n_train = int(len(instances)*NUM_TRAIN_INSTANCES)
    n_validation = int(len(instances)*NUM_VALIDATION_INSTANCES)
    n_test = int(len(instances)*NUM_TEST_INSTANCES)
    for fold in tqdm.trange(NUM_FOLDS, desc="Fold"):
        fold_instances = np.random.choice(
            len(instances), n_train + n_validation + n_test, replace=False)
        train = fold_instances[:n_train]
        validation = fold_instances[n_train:n_train + n_validation]
        test = fold_instances[n_train + n_validation:]

        for s, idxs in tqdm.tqdm(zip(["test", "train", "validation"], [test, train, validation]), total=3, desc="Split"):
            for i in tqdm.tqdm(idxs, desc=s):
                sim, tstep = instances[i]
                Oin, Oout, _0, _1 = process_instance(
                    read_instance(DATA_FOLDER, sim), tstep)

                Oin, Oout = normalise(Oin), normalise(Oout)
                Omerged = np.append(
                    Oin[np.newaxis, ...], Oout[np.newaxis, ...], axis=0)
                np.save(
                    "{}/{}/{}/{}.npy".format(DATASET_FOLDER,
                                            fold, s, i), Omerged
                )
            # end
        # end
    # end


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
