from __future__ import division
from __future__ import print_function

import itertools
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
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--runs', type=int, default=500,
                    help='Number of times to train and evaluate the model.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--layers_min', type=int, default=2,
                    help='Minimum number of layers to test with')
parser.add_argument('--layers_max', type=int, default=8,
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
  "RESK1": models.RESK1,
  "RESK2": models.RESK2,
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
    acc_threshold = (0.7879 + 0.7785)/2 * 0.9
    loss_threshold = (0.7375 + 0.7915)/2 * 1.1
elif args.dataset=="citeseer":
    acc_threshold = (0.6562 + 0.6451)/2 * 0.9
    loss_threshold = (1.1587 + 1.2430)/2 * 1.1
elif args.dataset=="pubmed":
    acc_threshold = (0.7739 + 0.7726)/2 * 0.9
    loss_threshold = (0.7068 + 0.7136)/2 * 1.1

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

    print("Optimization with model \"{model}\" on dataset \"{dataset}\" Finished!".format(model=m,dataset=args.dataset), flush=True)

    with open("{dataset}_{model}.tsv".format(dataset=args.dataset,model=m),"w") as f:
        print( "\t".join( [ "n_layers", "n_converged" , "n_bad", "converged_val_acc", "converged_val_acc_std", "converged_val_loss", "converged_val_loss_std", "layer_convergence",  "layer_convergence_std", "converged_test_acc", "converged_test_acc_std", "converged_test_loss", "converged_test_loss_std", "bad_val_acc", "bad_val_acc_std", "bad_val_loss", "bad_val_loss_std", "bad_epochs", "bad_epochs_std", "bad_test_acc", "bad_test_acc_std", "bad_test_loss", "bad_test_loss_std", ] ), file=f )

        model_data[m]["layer_converged_test_acc"] = np.zeros( args.layers_max )
        model_data[m]["layer_converged_test_acc_std"] = np.zeros( args.layers_max )
        model_data[m]["layer_bad_test_acc"] = np.zeros( args.layers_max )
        model_data[m]["layer_bad_test_acc_std"] = np.zeros( args.layers_max )
        model_data[m]["layer_convergence_ratio"] = np.zeros( args.layers_max )
        model_data[m]["layer_convergence_iters"] = np.zeros( args.layers_max )
        model_data[m]["layer_convergence_iters_std"] = np.zeros( args.layers_max )

        for nlayers in range(args.layers_min,args.layers_max):
            converged_idx = [ idx for idx, epochs in enumerate(model_data[m]["layer_convergence"][nlayers]) if epochs < args.epochs ]
            bad_idx = [ idx for idx, epochs in enumerate(model_data[m]["layer_convergence"][nlayers]) if epochs == args.epochs ]
            num_converged = len(converged_idx)
            num_bad = len(bad_idx)
            
            converged_val_acc = np.mean( model_data[m]["layer_val_acc"][nlayers,converged_idx,-1] )
            converged_val_acc_std = np.std( model_data[m]["layer_val_acc"][nlayers,converged_idx,-1] )
            converged_val_loss = np.mean( model_data[m]["layer_val_loss"][nlayers,converged_idx,-1] )
            converged_val_loss_std = np.std( model_data[m]["layer_val_loss"][nlayers,converged_idx,-1] )
            converged_epochs =  np.mean( model_data[m]["layer_convergence"][nlayers,converged_idx] )
            converged_epochs_std =  np.std( model_data[m]["layer_convergence"][nlayers,converged_idx] )
            converged_test_acc =  np.mean( model_data[m]["layer_test_acc"][nlayers,converged_idx] )
            converged_test_acc_std =  np.std( model_data[m]["layer_test_acc"][nlayers,converged_idx] )
            converged_test_loss =  np.mean( model_data[m]["layer_test_loss"][nlayers,converged_idx] )
            converged_test_loss_std =  np.std( model_data[m]["layer_test_loss"][nlayers,converged_idx] )
            
            bad_val_acc = np.mean( model_data[m]["layer_val_acc"][nlayers,bad_idx,-1] )
            bad_val_acc_std = np.std( model_data[m]["layer_val_acc"][nlayers,bad_idx,-1] )
            bad_val_loss = np.mean( model_data[m]["layer_val_loss"][nlayers,bad_idx,-1] )
            bad_val_loss_std = np.std( model_data[m]["layer_val_loss"][nlayers,bad_idx,-1] )
            bad_epochs =  np.mean( model_data[m]["layer_convergence"][nlayers,bad_idx] )
            bad_epochs_std =  np.std( model_data[m]["layer_convergence"][nlayers,bad_idx] )
            bad_test_acc =  np.mean( model_data[m]["layer_test_acc"][nlayers,bad_idx] )
            bad_test_acc_std =  np.std( model_data[m]["layer_test_acc"][nlayers,bad_idx] )
            bad_test_loss =  np.mean( model_data[m]["layer_test_loss"][nlayers,bad_idx] )
            bad_test_loss_std =  np.std( model_data[m]["layer_test_loss"][nlayers,bad_idx] )
            
            model_data[m]["layer_converged_test_acc"][nlayers] = converged_test_acc
            model_data[m]["layer_converged_test_acc_std"][nlayers] = converged_test_acc_std
            model_data[m]["layer_bad_test_acc"][nlayers] = bad_test_acc
            model_data[m]["layer_bad_test_acc_std"][nlayers] = bad_test_acc_std
            model_data[m]["layer_convergence_ratio"][nlayers] = num_converged/(num_converged+num_bad)
            model_data[m]["layer_convergence_iters"][nlayers] = converged_epochs
            model_data[m]["layer_convergence_iters_std"][nlayers] = converged_epochs_std
            
            print( "\t".join( [ str(nlayers), str(num_converged), str(num_bad), "{:.4f}".format(converged_val_acc), "{:.4f}".format(converged_val_acc_std), "{:.4f}".format(converged_val_loss), "{:.4f}".format(converged_val_loss_std), "{:.4f}".format(converged_epochs), "{:.4f}".format(converged_epochs_std), "{:.4f}".format(converged_test_acc), "{:.4f}".format(converged_test_acc_std), "{:.4f}".format(converged_test_loss), "{:.4f}".format(converged_test_loss_std), "{:.4f}".format(bad_val_acc), "{:.4f}".format(bad_val_acc_std), "{:.4f}".format(bad_val_loss), "{:.4f}".format(bad_val_loss_std), "{:.4f}".format(bad_epochs), "{:.4f}".format(bad_epochs_std), "{:.4f}".format(bad_test_acc), "{:.4f}".format(bad_test_acc_std), "{:.4f}".format(bad_test_loss), "{:.4f}".format(bad_test_loss_std), ] ), file=f )
        #end for nlayers
    #end with
#end for model

markers = ['+','x','*','.',',','o']
cmarkers = itertools.cycle(markers)
#colors = ["#6457a6","#dbd56e","#664e4c","#8c271e","#a4c2a8","#000000",]
colors = ["#6457a6ff","#dbd56eff","#664e4cff","#8c271eff","#a4c2a8ff","#000000ff",]
#colors = ['r','b','k','g','m','c']
ccolors = itertools.cycle(colors)
#colors_std = ['r','b','k','g','m','c']
colors_std = ["#6457a680","#dbd56e80","#664e4c80","#8c271e80","#a4c2a880","#00000080",]
ccolors_std = itertools.cycle(colors_std)


figformat = "pdf"

plt.title("Accuracy of Converged Models")
plt.ylim(-0.1,1.1)
plt.yticks( np.linspace(0,1,6) )
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data = [ (l,y) for l,y in zip(range(model_data[m]["min_layers"],model_data[m]["max_layers"]), model_data[m]["layer_converged_test_acc"][model_data[m]["min_layers"]:model_data[m]["max_layers"]]) if not np.isnan(y) ]
    plt.plot( [d[0] for d in data], [d[1] for d in data], label=m, marker=markers[i], color = colors[i] )
plt.legend()
plt.savefig("{dataset}_c_acc.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

plt.title("Accuracy of Nonconverged models")
plt.ylim(-0.1,1.1)
plt.yticks( np.linspace(0,1,6) )
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data = [ (l,y) for l,y in zip(range(model_data[m]["min_layers"],model_data[m]["max_layers"]), model_data[m]["layer_bad_test_acc"][model_data[m]["min_layers"]:model_data[m]["max_layers"]]) if not np.isnan(y) ]
    plt.plot( [d[0] for d in data], [d[1] for d in data], label=m, marker=markers[i], color = colors[i] )
plt.legend()
plt.savefig("{dataset}_nc_acc.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

plt.title("Ratio of Converged Models")
plt.ylim(-0.1,1.1)
plt.yticks( np.linspace(0,1,6) )
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data = [ (l,y) for l,y in zip(range(model_data[m]["min_layers"],model_data[m]["max_layers"]), model_data[m]["layer_convergence_ratio"][model_data[m]["min_layers"]:model_data[m]["max_layers"]]) if not np.isnan(y) ]
    plt.plot( [d[0] for d in data], [d[1] for d in data], label=m, marker=markers[i], color = colors[i] )
plt.legend()
plt.savefig("{dataset}_c_ratio.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

plt.title("Number of Iterations for converged models")
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data = [ (l,y) for l,y in zip(range(model_data[m]["min_layers"],model_data[m]["max_layers"]), model_data[m]["layer_convergence_iters"][model_data[m]["min_layers"]:model_data[m]["max_layers"]]) if not np.isnan(y) ]
    plt.plot( [d[0] for d in data], [d[1] for d in data], label=m, marker=markers[i], color = colors[i] )
plt.legend()
plt.savefig("{dataset}_c_iter.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

# Mean and Std plot

plt.title("Accuracy of Converged Models")
plt.ylim(-0.1,1.1)
plt.yticks( np.linspace(0,1,6) )
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data_mean_std = np.array(
      [
        [layers,mean,std if np.isfinite(std) else 0] for layers,mean,std in zip(
          range(model_data[m]["min_layers"],model_data[m]["max_layers"]),
          model_data[m]["layer_converged_test_acc"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
          model_data[m]["layer_converged_test_acc_std"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
        ) if not np.isnan(mean)
      ]
    )
    if data_mean_std.shape[0] != 0:
        plot_mean_and_std( data_mean_std[:,0],data_mean_std[:,1], data_mean_std[:,2], color_mean=colors[i], marker_mean=markers[i], color_shading=colors[i], label=m )
plt.legend()
plt.savefig("{dataset}_c_acc_std.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

plt.title("Accuracy of Nonconverged models")
plt.ylim(-0.1,1.1)
plt.yticks( np.linspace(0,1,6) )
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data_mean_std = np.array(
      [
        [layers,mean,std if np.isfinite(std) else 0] for layers,mean,std in zip(
          range(model_data[m]["min_layers"],model_data[m]["max_layers"]),
          model_data[m]["layer_bad_test_acc"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
          model_data[m]["layer_bad_test_acc_std"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
        ) if not np.isnan(mean)
      ]
    )
    if data_mean_std.shape[0] != 0:
        plot_mean_and_std( data_mean_std[:,0],data_mean_std[:,1], data_mean_std[:,2], color_mean=colors[i], marker_mean=markers[i], color_shading=colors[i], label=m )
plt.legend()
plt.savefig("{dataset}_nc_acc_std.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()

plt.title("Number of Iterations for converged models")
plt.xlim(1,args.layers_max)
plt.xticks( range(1,args.layers_max+1), [str(x) for x in range(1,args.layers_max)] )
for i, m in enumerate(model_dict):
    data_mean_std = np.array(
      [
        [layers,mean,std if np.isfinite(std) else 0] for layers,mean,std in zip(
          range(model_data[m]["min_layers"],model_data[m]["max_layers"]),
          model_data[m]["layer_convergence_iters"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
          model_data[m]["layer_convergence_iters_std"][model_data[m]["min_layers"]:model_data[m]["max_layers"]],
        ) if not np.isnan(mean)
      ]
    )
    if data_mean_std.shape[0] != 0:
        plot_mean_and_std( data_mean_std[:,0],data_mean_std[:,1], data_mean_std[:,2], color_mean=colors[i], marker_mean=markers[i], color_shading=colors[i], label=m )
plt.legend()
plt.savefig("{dataset}_c_iter_std.{figformat}".format(dataset=args.dataset,figformat=figformat))
plt.close()
