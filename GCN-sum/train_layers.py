from __future__ import division
from __future__ import print_function

import itertools
import pickle
import time
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data_new as load_data, accuracy, count_params, plot_mean_and_std
import models

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--runs', type=int, default=500,
                    help='Number of times to train and evaluate the model.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--layers_min', type=int, default=3,
                    help='Minimum number of layers to test with')
parser.add_argument('--layers_max', type=int, default=5,
                    help='Maximum number of layers to test with')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', choices=["cora", "citeseer", "pubmed"], default="cora",
                    help='Which dataset to use')
parser.add_argument('--early_stopping_epochs', type=int, default=10,
                    help='Number of epochs to evaluate early stopping on.')
parser.add_argument('--early_stopping_threshold', type=float, default=1e-10,
                    help='Minimum decrease in validation loss over last early_stopping_epochs.')
model_dict = {
  "GCNK": models.GCNK,
#  "GCNKnorm": models.GCNKnorm,
  "RESK1": models.RESK1,
  "RESK2": models.RESK2,
  "RESK1norm": models.RESK1norm,
  "RESK2norm": models.RESK2norm,
#  "ODEK1": models.ODEK1,
#  "ODEK2": models.ODEK2,
}

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert( args.layers_min < args.layers_max )
args.layers_max += 1

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(model, optimizer, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if False:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()


def test(model, optimizer):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if False:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()

if args.dataset=="cora":
    acc_threshold = 0.7782 * 0.9
    loss_threshold = 0.7929 * 1.1
elif args.dataset=="citeseer":
    acc_threshold = 0.6443 * 0.9
    loss_threshold = 1.2454 * 1.1
elif args.dataset=="pubmed":
    acc_threshold = 0.7726 * 0.9
    loss_threshold = 0.7136 * 1.1

model_data = {}
for m in model_dict:
    model_data[m] = {}
    GCN = model_dict[m]
    
    model_data[m]["layer_val_acc"] = np.zeros( [args.layers_max, args.runs, args.epochs] )
    model_data[m]["layer_val_loss"] = np.zeros( [args.layers_max, args.runs, args.epochs] )
    model_data[m]["layer_convergence"] = args.epochs * np.ones( [args.layers_max, args.runs])
    model_data[m]["layer_test_acc"] = np.zeros( [args.layers_max, args.runs] )
    model_data[m]["layer_test_loss"] = np.zeros( [args.layers_max, args.runs] )

    model_data[m]["min_layers"] = args.layers_min
    model_data[m]["max_layers"] = args.layers_max
    for nlayers in range(args.layers_min,args.layers_max):
        for run in range(args.runs):
            # Model and optimizer
            try:
                model = GCN(nfeat=features.shape[1],
                            nhid=args.hidden,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout,
                            nlayers=nlayers
                            )
            except ValueError:
                model_data[m]["min_layers"] = nlayers+1
                # Can't build a res network with that many blocks
                continue
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)

            if args.cuda:
                model.cuda()
            
            for epoch in range(args.epochs):
                epoch_val_loss, epoch_val_acc = train(model, optimizer, epoch)
                model_data[m]["layer_val_loss"][nlayers,run,epoch] = epoch_val_loss
                model_data[m]["layer_val_acc"][nlayers,run,epoch] = epoch_val_acc
                if epoch_val_acc > acc_threshold and epoch_val_loss < loss_threshold:
                    model_data[m]["layer_convergence"][nlayers,run] = epoch
                    model_data[m]["layer_val_loss"][nlayers,run,epoch:] = model_data[m]["layer_val_loss"][nlayers,run,epoch-1]
                    model_data[m]["layer_val_acc"][nlayers,run,epoch:] = model_data[m]["layer_val_acc"][nlayers,run,epoch-1]
                    break
            
            run_test_loss, run_test_acc = test(model, optimizer)
            model_data[m]["layer_test_loss"][nlayers,run] = run_test_loss
            model_data[m]["layer_test_acc"][nlayers,run] = run_test_acc
            
            print( "{nlayers} layers's run #{run} Test -- epochs: {epochs:d} acc: {acc:.2f}%".format( nlayers=nlayers, run=run, epochs=int(model_data[m]["layer_convergence"][nlayers,run]), acc=100*model_data[m]["layer_test_acc"][nlayers,run] ), flush=True )
        #end for run
        if model_data[m]["min_layers"] <= nlayers and np.mean(model_data[m]["layer_test_acc"][nlayers,:]) < 0.5:
            model_data[m]["max_layers"] = nlayers
            break
        #end if
    #end for layers
    print("Optimization with model \"{model}\" on dataset \"{dataset}\" Finished!".format(model=m,dataset=args.dataset), flush=True)

    with open("{dataset}_{model}.pickle".format(dataset=args.dataset,model=m),"wb") as f:
        pickle.dump( model_data[m], f, protocol=pickle.HIGHEST_PROTOCOL )
    #end with
#end for
