import os
import tqdm

from pprint import pprint as pp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import fire

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


def train(
        train=False,
        test=False,
        num_of_bodies=6,
        model="IN",
        num_epochs=2000,
        batch_size=1000, # Higher since it only occupies about 1gb of VRAM
        num_folds=10
    ):
    PERCENTILES_FNAME = "./dataset/percentiles.npy"
    DATASET_FOLDER = "./dataset/{}".format(num_of_bodies)
    OUTPUT_FOLDER = "./output"
    MODEL_FOLDER = "./model"
    O_SHAPE = 5  # vx,vy,px,py,m
    LEARNING_RATE = 0.001
    LR_DECAY = 0.8
    LR_DECAY_WINDOW = 40
    VAL_LOSS_HIGH = 40.0  # High loss to initialise the decay window
    L2_NORM = 5e-4  # Not provided in paper, took from our other experiments
    PREDICTED_VALUES = 2  # vx, vy
    NOISE_STD = 0.05  # We use 0.05 std instead of 0.05 of the dataset's std
    NOISE_PROPORTION_MAX = 0.2
    NOISE_EPOCH_START_DECAY = 50
    NOISE_EPOCH_STOP_DECAY = 250
    USE_CUDA = True
    VALID_LOG_FNAME = "log.{model_name}.{fold}.valid"
    TEST_LOG_FNAME = "log.{model_name}.test"
    use_cuda = USE_CUDA and torch.cuda.is_available()
    
    model_name=model
    model=None
    if model_name=="IN":
        Model = IN
    elif model_name=="IN_ODE":
        Model = IN_ODE
    else:
        raise ValueError("Model must be either IN OR IN_ODE")
    #end

    dataset = np.load("{}/dataset.npy".format(DATASET_FOLDER)
                      ).astype(np.float32)
                      
    value_percentiles = np.load(PERCENTILES_FNAME).astype(np.float32)[:,:PREDICTED_VALUES]
    pytvalue_percentiles = torch.tensor(value_percentiles)
    if use_cuda:
        pytvalue_percentiles = pytvalue_percentiles.gpu()
    denormalise = lambda x: (((x+1)/2)*(value_percentiles[2]-value_percentiles[0]))+value_percentiles[1]
    pytdenormalise = lambda x: (((x+1)/2)*(pytvalue_percentiles[2]-pytvalue_percentiles[0]))+pytvalue_percentiles[1]

    if train:
        test_log_file = open(TEST_LOG_FNAME.format(model_name=model_name),"w")
        tqdm.tqdm.write("Train procedure...")
        for fold in tqdm.trange(num_folds, desc="Fold"):
            valid_log_file = open(VALID_LOG_FNAME.format(model_name=model_name, fold=fold),"w")
            current_lr = LEARNING_RATE
            model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
            if use_cuda:
                model = model.cuda()
            optimizer = optim.Adam(model.parameters(),
                                   lr=current_lr, weight_decay=L2_NORM)
            train = np.load("{}/{}.train.npy".format(DATASET_FOLDER, fold))
            test = np.load("{}/{}.test.npy".format(DATASET_FOLDER, fold))
            validation = np.load(
                "{}/{}.validation.npy".format(DATASET_FOLDER, fold))
            validation_sch = np.ones(LR_DECAY_WINDOW) * VAL_LOSS_HIGH

            for epoch in tqdm.trange(num_epochs, desc="Epoch"):
                model.train()
                for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, train, batch_size)), total=train.shape[0]/batch_size, desc="Batch Train"):
                    # Random noise schedule
                    if False and epoch < NOISE_EPOCH_STOP_DECAY:
                        bOin, bOout, bMsrc, bMtgt, n_list, m_list = batch
                        noise = np.random.normal(
                            0, NOISE_STD, size=np.prod(bOin.shape))
                        # decay the proportion
                        proportion = NOISE_PROPORTION_MAX
                        if epoch > NOISE_EPOCH_START_DECAY:
                            proportion *= 1 - \
                                ((NOISE_EPOCH_STOP_DECAY-epoch) /
                                 (NOISE_EPOCH_STOP_DECAY-NOISE_EPOCH_START_DECAY))
                        # end if
                        idx = np.random.choice(noise.shape[0], replace=False, size=int(
                            noise.shape[0] * (1-proportion)))
                        noise[idx] = 0
                        bOin += noise.reshape(bOin.shape)
                        batch = (bOin, bOout, bMsrc, bMtgt, n_list, m_list)
                    # end if
                    bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(lambda x: torch.tensor(x.astype(
                        np.float32)) if not type(x) == list else torch.tensor(x, dtype=torch.float), batch)
                    if use_cuda:
                        bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(
                            lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt, n_list, m_list))
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    optimizer.zero_grad()
                    loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                    loss.backward()
                    optimizer.step()
                # end for

                if validation.shape[0]>0:
                    model.eval()
                    val_loss = []
                    val_loss_unnorm = []
                    for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, validation, batch_size)), total=validation.shape[0]/batch_size, desc="Batch Valid"):
                        bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(lambda x: torch.tensor(x.astype(
                            np.float32)) if not type(x) == list else torch.tensor(x, dtype=torch.float), batch)
                        if use_cuda:
                            bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(
                                lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt, n_list, m_list))
                        with torch.no_grad():
                            Pred = model(bOin, None, None, bMsrc, bMtgt)
                            loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                            unnorm_loss = F.mse_loss(pytdenormalise(Pred), pytdenormalise(bOout[:, :PREDICTED_VALUES]))
                        # end with
                        val_loss.append(loss.cpu().item())
                        val_loss_unnorm.append(unnorm_loss.cpu().item())
                    # end for

                    val_loss = np.mean(val_loss)
                    val_loss_unnorm = np.mean(val_loss_unnorm)
                    print("{batch},{loss:f},{unnorm_loss}".format(batch=b,loss=valid_loss,unnorm_loss=test_loss), file=valid_log_file)
                    validation_sch[:-1] = validation_sch[1:]
                    validation_sch[-1] = val_loss
                    if np.sum(validation_sch[1:]-validation_sch[:-1]) > 0:
                        current_lr *= LR_DECAY
                        optimizer = optim.Adam(model.parameters(),
                                               lr=current_lr, weight_decay=L2_NORM)
                    # end if
                #end if
            # end for
            valid_log_file.close()

            model.eval()
            test_loss = []
            test_loss_unnorm = []
            for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, test, batch_size)), total=test.shape[0]/batch_size, desc="Test"):
                bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(lambda x: torch.tensor(x.astype(
                    np.float32)) if not type(x) == list else torch.tensor(x, dtype=torch.float), batch)
                if use_cuda:
                    bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(
                        lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt, n_list, m_list))
                with torch.no_grad():
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                    unnorm_loss = F.mse_loss(pytdenormalise(Pred), pytdenormalise(bOout[:, :PREDICTED_VALUES]))
                # end with
                test_loss.append(loss.cpu().item())
                test_loss_unnorm.append(unnorm_loss.cpu().item())
            # end for

            test_loss = np.mean(test_loss)
            test_loss_unnorm = np.mean(test_loss_unnorm)
            tqdm.tqdm.write("Test_loss: " + str(test_loss))
            print("{fold},{loss:f},{unnorm_loss}".format(fold=fold,loss=test_loss,unnorm_loss=test_loss_unnorm), file=test_log_file)

            SAVE_PATH = "{}/{model_name}_{fold}_{epoch}".format(MODEL_FOLDER, model_name=model_name, fold=fold, epoch=epoch)
            torch.save(model.state_dict(), SAVE_PATH)
            
        # end for
        test_log_file.close()
    elif test:
        tqdm.tqdm.write("Test procedure...")

        # Load model
        model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
        LOAD_PATH = sorted(x for x in os.listdir(MODEL_FOLDER) if model_name in x)[-1]
        model.load_state_dict(torch.load(LOAD_PATH))

        # Configure BATCH_SIZE
        batch_size = 999

        for fold in tqdm.trange(num_folds, desc="Fold"):
            test = np.load("{}/{}.test.npy".format(DATASET_FOLDER, fold))
            model.eval()
            test_loss = []
            test_loss_unnorm = []
            Pred_save = np.array([])
            for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, test, batch_size, shuffle=False)), total=test.shape[0]/batch_size, desc="Test"):
                bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(lambda x: torch.tensor(x.astype(
                    np.float32)) if not type(x) == list else torch.tensor(x, dtype=torch.float), batch)
                if use_cuda:
                    bOin, bOout, bMsrc, bMtgt, n_list, m_list = map(
                        lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt, n_list, m_list))
                with torch.no_grad():
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                    unnorm_loss = F.mse_loss(pytdenormalise(Pred), pytdenormalise(bOout[:, :PREDICTED_VALUES]))
                # end with

                test_loss.append(loss.cpu().item())
                test_loss_unnorm.append(unnorm_loss.cpu().item())
                Pred_save.append(denormalise(Pred.cpu().detach().numpy()))
            # end for

            test_loss = np.mean(test_loss)
            test_loss_unnorm = np.mean(test_loss_unnorm)
            tqdm.tqdm.write("| Loss | " + str(test_loss) + "| Denormalised Loss | " + str(test_loss_unnorm))

            # Save bOout to output
            np.save("{}/{}.npy".format(OUTPUT_FOLDER, fold))


if __name__ == "__main__":
    fire.Fire(train)
