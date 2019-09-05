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
        rollout=False,
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
                      
    value_percentiles = np.load(PERCENTILES_FNAME).astype(np.float32)
    pytvalue_percentiles = torch.tensor(value_percentiles)
    velocity_percentiles = value_percentiles[:,:PREDICTED_VALUES]
    pytvvelocity_percentiles = torch.tensor(velocity_percentiles)
    if use_cuda:
        pytvalue_percentiles = pytvalue_percentiles.cuda()
        pytvvelocity_percentiles = pytvvelocity_percentiles.cuda()
    denormalise_v = lambda x: (((x+1)/2)*(velocity_percentiles[2]-velocity_percentiles[0]))+velocity_percentiles[1]
    pytdenormalise_v = lambda x: (((x+1)/2)*(pytvvelocity_percentiles[2]-pytvvelocity_percentiles[0]))+pytvvelocity_percentiles[1]
    denormalise = lambda x: (((x+1)/2)*(value_percentiles[2]-value_percentiles[0]))+value_percentiles[1]
    pytdenormalise = lambda x: (((x+1)/2)*(pytvalue_percentiles[2]-pytvalue_percentiles[0]))+pytvalue_percentiles[1]
    pytnormalise = lambda x: ((2*((x-pytvalue_percentiles[1])/(pytvalue_percentiles[2]-pytvalue_percentiles[0])))-1)

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
                        batch = [bOin, bOout, bMsrc, bMtgt, n_list, m_list]
                    # end if
                    bOin, bOout, bMsrc, bMtgt = map(lambda x: torch.tensor(x.astype(
                        np.float32)), batch[:-2])
                    n_list, m_list = batch[-2:]
                    if use_cuda:
                        bOin, bOout, bMsrc, bMtgt = map(
                            lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt))
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
                        bOin, bOout, bMsrc, bMtgt = map(lambda x: torch.tensor(x.astype(
                            np.float32)), batch[:-2])
                        n_list, m_list = batch[-2:]
                        with torch.no_grad():
                            Pred = model(bOin, None, None, bMsrc, bMtgt)
                            loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                            unnorm_loss = F.mse_loss(pytdenormalise_v(Pred), pytdenormalise_v(bOout[:, :PREDICTED_VALUES]))
                        # end with
                        val_loss.append(loss.cpu().item())
                        val_loss_unnorm.append(unnorm_loss.cpu().item())
                    # end for

                    val_loss = np.mean(val_loss)
                    val_loss_unnorm = np.mean(val_loss_unnorm)
                    print("{batch},{loss:f},{unnorm_loss}".format(batch=b,loss=valid_loss,unnorm_loss=test_loss), file=valid_log_file, flush=True)
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
                bOin, bOout, bMsrc, bMtgt = map(lambda x: torch.tensor(x.astype(
                    np.float32)), batch[:-2])
                n_list, m_list = batch[-2:]
                if use_cuda:
                    bOin, bOout, bMsrc, bMtgt = map(
                        lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt))
                with torch.no_grad():
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                    unnorm_loss = F.mse_loss(pytdenormalise_v(Pred),pytdenormalise_v(bOout[:, :PREDICTED_VALUES]))
                # end with
                test_loss.append(loss.cpu().item())
                test_loss_unnorm.append(unnorm_loss.cpu().item())
            # end for

            test_loss = np.mean(test_loss)
            test_loss_unnorm = np.mean(test_loss_unnorm)
            print("{fold},{loss:f},{unnorm_loss}".format(fold=fold,loss=test_loss,unnorm_loss=test_loss_unnorm), file=test_log_file, flush=True)
            tqdm.tqdm.write("Test_loss: " + str(test_loss) + " Denormalised Test Loss: "+ str(test_loss_unnorm))

            SAVE_PATH = "{}/{model_name}_{fold}_{epoch}".format(MODEL_FOLDER, model_name=model_name, fold=fold, epoch=epoch)
            torch.save(model.state_dict(), SAVE_PATH)
            
        # end for
        test_log_file.close()
    elif test:
        tqdm.tqdm.write("Test procedure...")
        test_log_file = open(TEST_LOG_FNAME.format(model_name=model_name),"w")
        tqdm.tqdm.write("Train procedure...")
        for fold in tqdm.trange(num_folds, desc="Fold"):
            # Load model
            model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
            if use_cuda:
                model = model.cuda()
            model_prefix = "{model_name}_{fold}_".format(model_name=model_name, fold=fold)
            saved_files = sorted(x for x in os.listdir(MODEL_FOLDER) if model_prefix in x )
            
            model.load_state_dict(torch.load("{}/{}".format(MODEL_FOLDER,saved_files[-1])))
            model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
            if use_cuda:
                model = model.cuda()
            test = np.load("{}/{}.test.npy".format(DATASET_FOLDER, fold))

            model.eval()
            test_loss = []
            test_loss_unnorm = []
            for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, test, batch_size)), total=test.shape[0]/batch_size, desc="Test"):
                bOin, bOout, bMsrc, bMtgt = map(lambda x: torch.tensor(x.astype(
                    np.float32)), batch[:-2])
                n_list, m_list = batch[-2:]
                if use_cuda:
                    bOin, bOout, bMsrc, bMtgt = map(
                        lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt))
                with torch.no_grad():
                    Pred = model(bOin, None, None, bMsrc, bMtgt)
                    loss = F.mse_loss(Pred, bOout[:, :PREDICTED_VALUES])
                    unnorm_loss = F.mse_loss(pytdenormalise_v(Pred),pytdenormalise_v(bOout[:, :PREDICTED_VALUES]))
                # end with
                test_loss.append(loss.cpu().item())
                test_loss_unnorm.append(unnorm_loss.cpu().item())
            # end for

            test_loss = np.mean(test_loss)
            test_loss_unnorm = np.mean(test_loss_unnorm)
            print("{fold},{loss:f},{unnorm_loss}".format(fold=fold,loss=test_loss,unnorm_loss=test_loss_unnorm), file=test_log_file, flush=True)
            tqdm.tqdm.write("Test_loss: " + str(test_loss) + " Denormalised Test Loss: "+ str(test_loss_unnorm))

            SAVE_PATH = "{}/{model_name}_{fold}_{epoch}".format(MODEL_FOLDER, model_name=model_name, fold=fold, epoch=epoch)
            torch.save(model.state_dict(), SAVE_PATH)
            
        # end for
        test_log_file.close()
    elif rollout:
        tqdm.tqdm.write("Rollout procedure...")

        # Load model
        model = Model(O_SHAPE, 0, 0, PREDICTED_VALUES)
        if use_cuda:
            model = model.cuda()
        saved_files = sorted(x for x in os.listdir(MODEL_FOLDER) if (("ODE" in model_name) == ("ODE" in x)) )
        model.load_state_dict(torch.load("{}/{}".format(MODEL_FOLDER,saved_files[-1])))

        # Configure BATCH_SIZE
        batch_size = 999

        for fold in tqdm.trange(num_folds, desc="Fold"):
            test = np.load("{}/{}.test.npy".format(DATASET_FOLDER, fold))
            model.eval()
            test_loss = []
            test_loss_unnorm = []
            with torch.no_grad():
                for b, batch in tqdm.tqdm(enumerate(get_epoch(dataset, test, batch_size, shuffle=False)), total=test.shape[0]/batch_size, desc="Test"):
                    if b not in [2,3]:
                        continue
                    O_save = torch.tensor(np.zeros([batch_size+1,num_of_bodies,O_SHAPE], dtype=np.float32))
                    
                    bOin, bOout, bMsrc, bMtgt = map(lambda x: torch.tensor(x.astype(
                        np.float32)), batch[:-2])
                    n_list, m_list = batch[-2:]
                    bMsrc = bMsrc[:n_list[0],:m_list[0]]
                    bMtgt = bMtgt[:n_list[0],:m_list[0]]
                    if use_cuda:
                        bOin, bOout, bMsrc, bMtgt = map(
                            lambda x: x.cuda(), (bOin, bOout, bMsrc, bMtgt))
                        O_save = O_save.cuda()
                    #end if
                    O_save[0] = bOin[:n_list[0]]
                    for i in range(1,batch_size+1):
                            Pred = model(O_save[i-1], None, None, bMsrc, bMtgt)
                            denorm_pred = np.empty_like(Pred.cpu().detach().numpy())
                            denorm_pred[...] = denormalise_v(Pred.cpu().detach().numpy())
                            denorm_pred = torch.tensor(denorm_pred)
                            denorm_last = np.empty_like(O_save[i-1].cpu().detach().numpy())
                            denorm_last[...] = denormalise(O_save[i-1].cpu().detach().numpy())
                            denorm_last = torch.tensor(denorm_last)
                            if use_cuda:
                                denorm_pred = denorm_pred.cuda()
                                denorm_last = denorm_last.cuda()
                            denorm_last[:,PREDICTED_VALUES:PREDICTED_VALUES+2] += (denorm_last[:,:PREDICTED_VALUES]+denorm_pred)/2*0.01
                            denorm_last[:,:PREDICTED_VALUES] = denorm_pred
                            O_save[i,:,:PREDICTED_VALUES] = pytnormalise(denorm_last)[:,:PREDICTED_VALUES]
                            O_save[i,:,PREDICTED_VALUES:PREDICTED_VALUES+2] = pytnormalise(denorm_last)[:,PREDICTED_VALUES:PREDICTED_VALUES+2]
                            O_save[i,:,-1] = O_save[i-1,:,-1]
                    #end for
                    # Save bOout to output
                    np.save("{}/{fold}_{batch}.npy".format(OUTPUT_FOLDER, fold=fold, batch=b), denormalise(O_save.cpu().detach().numpy()))
                    #exit()
                # end for
            # end with
        #end for
    else:
        print("Provide one of: --train, --test, --rollout")
    #end if



if __name__ == "__main__":
    fire.Fire(train)
