import pickle
import itertools
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from utils import plot_mean_and_std
import models

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--runs', type=int, default=500,
                    help='Number of times to train and evaluate the model.')
parser.add_argument('--layers_min', type=int, default=2,
                    help='Minimum number of layers to test with')
parser.add_argument('--layers_max', type=int, default=4,
                    help='Maximum number of layers to test with')
parser.add_argument('--dataset', choices=["cora", "citeseer", "pubmed"], default="cora",
                    help='Which dataset to use')

model_dict = {
  "GCNK": None,
  "RESK1": None,
  "RESK2": None,
}

args = parser.parse_args()
assert( args.layers_min < args.layers_max )
args.layers_max += 1

model_data = {}
for m in model_dict:
    model_data[m] = {}
    
    with open("{dataset}_{model}.pickle".format(dataset=args.dataset,model=m),"rb") as f:
        model_data[m] = pickle.load(f)
    #end with

    #print( "\t".join( [ "n_layers", "n_converged" , "n_bad", "converged_val_acc", "converged_val_acc_std", "converged_val_loss", "converged_val_loss_std", "layer_convergence",  "layer_convergence_std", "converged_test_acc", "converged_test_acc_std", "converged_test_loss", "converged_test_loss_std", "bad_val_acc", "bad_val_acc_std", "bad_val_loss", "bad_val_loss_std", "bad_epochs", "bad_epochs_std", "bad_test_acc", "bad_test_acc_std", "bad_test_loss", "bad_test_loss_std", ] ), file=f )
    
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
        
        #print( "\t".join( [ str(nlayers), str(num_converged), str(num_bad), "{:.4f}".format(converged_val_acc), "{:.4f}".format(converged_val_acc_std), "{:.4f}".format(converged_val_loss), "{:.4f}".format(converged_val_loss_std), "{:.4f}".format(converged_epochs), "{:.4f}".format(converged_epochs_std), "{:.4f}".format(converged_test_acc), "{:.4f}".format(converged_test_acc_std), "{:.4f}".format(converged_test_loss), "{:.4f}".format(converged_test_loss_std), "{:.4f}".format(bad_val_acc), "{:.4f}".format(bad_val_acc_std), "{:.4f}".format(bad_val_loss), "{:.4f}".format(bad_val_loss_std), "{:.4f}".format(bad_epochs), "{:.4f}".format(bad_epochs_std), "{:.4f}".format(bad_test_acc), "{:.4f}".format(bad_test_acc_std), "{:.4f}".format(bad_test_loss), "{:.4f}".format(bad_test_loss_std), ] ), file=f )
    #end for nlayers
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
