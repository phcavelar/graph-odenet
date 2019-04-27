from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data_new as load_data, accuracy, count_params
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
parser.add_argument('--runs', type=int, default=1,
                    help='Number of times to train and evaluate the model.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', choices=["cora", "citeseer", "pubmed"], default="cora",
                    help='Which dataset to use')
parser.add_argument('--model', choices=["gcn2", "gcn3", "gcn3norm", "res3", "ode3", "res3norm", "res3fullnorm", "ode3norm"], default="res3",
                    help='Which model to train')
model_dict = {"GCN2": models.GCN, "GCN3": models.GCN3, "GCN3NORM": models.GCN3, "RES3": models.RGCN3, "ODE3": models.ODEGCN3, "RES3NORM": models.RGCN3norm, "RES3FULLNORM": models.RGCN3fullnorm, "ODE3NORM": models.ODEGCN3fullnorm}

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

GCN = model_dict[args.model.upper()]

if args.runs == 1:
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
    model.nfe = 0
    
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    is_ode = "ode" in args.model
    if is_ode:
      nfe_forward = model.nfe
      model.nfe = 0
    
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    if is_ode:
        nfe_backward = model.nfe
        model.nfe = 0

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if args.runs == 1:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t),
              "" if not is_ode else 'nfe_f: {}'.format(nfe_forward),
              "" if not is_ode else 'nfe_b: {}'.format(nfe_backward),
              )


def test(model, optimizer):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.runs == 1:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()


# Train model
total_loss, total_acc, total_time = 0,0,0

for run in range(args.runs):
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
    try:
      run_tstart = time.time()
      for epoch in range(args.epochs):
          train(model, optimizer, epoch)
          
      run_time = time.time() - run_tstart
      run_loss, run_acc = test(model, optimizer)
          
      if args.runs>1:
        print( "Run #{run} Test -- time: {time}s acc: {acc:.2f}%".format( run=run, time=run_time, acc=100*run_acc ), flush=True )
    except KeyboardInterrupt:
      args.runs = run
      break
      
    total_loss += run_loss
    total_acc += run_acc
    total_time += run_time

total_loss, total_acc, total_time = map(lambda x: x/args.runs, [total_loss, total_acc, total_time])

print("Optimization on dataset \"{dataset}\" Finished!".format(dataset=args.dataset))
print("#Parameters: {param_count}".format(param_count=count_params(model)))
print("Average time elapsed: {:.4f}s".format(total_time))

# Testing
print("Test set results:",
      "avg loss= {:.4f}".format(total_loss),
      "avg accuracy= {:.4f}".format(total_acc))
