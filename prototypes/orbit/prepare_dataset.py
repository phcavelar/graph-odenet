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


def get_epoch(dataset, indexes, batch_size=100, shuffle=True):
    if shuffle:
        np.random.shuffle(indexes)

    epoch = dataset[indexes]
    num_instances = epoch.shape[0]

    for i in range(0, num_instances, batch_size):
        yield gen_batch(epoch[i:i+batch_size])

    if i < num_instances:
        yield gen_batch(epoch[i:])
# end get_epoch


def gen_batch(batch):
    batch_size = batch.shape[0]
    batch_n = sum((batch[i, 0].shape[0] for i in range(batch_size)))
    batch_m = sum((batch[i, 0].shape[0]*batch[i, 0].shape[0] -
                   batch[i, 0].shape[0] for i in range(batch_size)))
    float_dtype = batch.dtype
    O_shape = batch.shape[-1]
    bOin = np.zeros([batch_n, O_shape], float_dtype)
    bOout = np.zeros([batch_n, O_shape], float_dtype)
    bMsrc = np.zeros([batch_n, batch_m], float_dtype)
    bMtgt = np.zeros([batch_n, batch_m], float_dtype)
    n_list = []
    m_list = []
    n_acc = 0
    m_acc = 0

    for i in range(batch_size):
        Oin, Oout = batch[i, 0], batch[i, 1]
        n = Oin.shape[0]

        bOin[n_acc:n_acc+n, :] = Oin[:, :]
        bOout[n_acc:n_acc+n, :] = Oout[:, :]

        Adj_matrix = np.ones([n, n]) - np.eye(n)
        relations = [(src, tgt) for src in range(n)
                     for tgt in range(n) if Adj_matrix[src, tgt] != 0]

        m = len(relations)

        for r, (s, t) in enumerate(relations):
            bMsrc[n+s, m+r] = 1
            bMtgt[n+t, m+r] = 1
        # end for

        n_list.append(n)
        m_list.append(m)
        n_acc+=n
        m_acc+=m
    # end for
    
    return bOin, bOout, bMsrc, bMtgt, n_list, m_list
# end gen_batch


DEFAULT_NUM_OF_BODIES = 6


def prepare_dataset(num_of_bodies=DEFAULT_NUM_OF_BODIES, train_pct=.5, test_pct=.5, val_pct=.0, max_timesteps=1000, num_folds=10):
    DATA_FOLDER = "./data/{}".format(num_of_bodies)
    DATASET_FOLDER = "./dataset/{}".format(num_of_bodies)
    PERCENTILES_FILE = "./dataset/percentiles.npy"
    assert sum(train_pct, test_pct,
               val_pct) <= 1, "Total percentage must sum to at most 1"

    print("Cleaning and preparing dataset folders")
    if os.path.isdir(DATASET_FOLDER):
        shutil.rmtree(DATASET_FOLDER)

    os.mkdir(DATASET_FOLDER)

    simulations = [x for x in sorted(
        os.listdir(DATA_FOLDER)) if x != ".gitkeep"]
    values_size = len(simulations)*max_timesteps
    Oin = get_O(read_instance(DATA_FOLDER, simulations[0]), 0)
    input_shape = Oin.shape
    inputs = np.zeros([values_size, *input_shape])
    vidx = 0
    for sim in tqdm.tqdm(simulations):
        sim_instance = read_instance(DATA_FOLDER, sim)
        for t in tqdm.trange(max_timesteps):
            Oin = get_O(sim_instance, t)
            inputs[vidx, ...] = Oin[...]
            vidx += 1
        # end for
    # end for
    percentiles = [5, 50, 95]
    value_percentiles = np.zeros([len(percentiles), input_shape[-1]])

    if num_of_bodies != DEFAULT_NUM_OF_BODIES:  # Load percentiles
        value_percentiles = np.load(PERCENTILES_FILE)
    else:  # Compute percentiles and save them
        for attridx in range(input_shape[-1]):
            value_percentiles[:, attridx] = np.percentile(
                inputs[:, :, attridx], percentiles)
        np.save(PERCENTILES_FILE, value_percentiles)

    np.save(
        "{}/normvals.npy".format(DATASET_FOLDER), value_percentiles
    )

    values_size = len(simulations)*(max_timesteps-1)
    dataset = np.zeros([values_size, 2, *input_shape])
    vidx = 0
    for sim in tqdm.tqdm(simulations):
        sim_instance = read_instance(DATA_FOLDER, sim)
        for t in tqdm.trange(max_timesteps-1):
            Oin, Oout = get_O(sim_instance, t), get_O(sim_instance, t+1)
            Oin, Oout = map(lambda x: (
                2 * ((x - value_percentiles[1]) / (value_percentiles[2] - value_percentiles[0]))) - 1, [Oin, Oout])
            dataset[vidx, 0, ...] = Oin[...]
            dataset[vidx, 1, ...] = Oout[...]
            vidx += 1
        # end for
    # end for

    np.save(
        "{}/dataset.npy".format(DATASET_FOLDER), dataset
    )

    n_train = int(values_size*train_pct)
    n_validation = int(values_size*val_pct)
    n_test = int(values_size*test_pct)

    for fold in tqdm.trange(num_folds, desc="Fold"):
        fold_instances = np.random.choice(
            values_size, n_train + n_validation + n_test, replace=False)
        train = fold_instances[:n_train]
        validation = fold_instances[n_train:n_train + n_validation]
        test = fold_instances[n_train + n_validation:]

        np.save("{}/{}.train.npy".format(DATASET_FOLDER, fold), train)
        np.save("{}/{}.validation.npy".format(DATASET_FOLDER, fold), validation)
        np.save("{}/{}.test.npy".format(DATASET_FOLDER, fold), test)
    # end


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
