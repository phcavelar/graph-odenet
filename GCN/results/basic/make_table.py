import sys
import numpy as np

def get_stats( fname ):
  with open(fname) as f:
    vals = []
    for l in f:
      if "Run" in l:
        vals.append(float(l[-7:-2]))
      if "Average" in l:
        avg_time = float(l[:-2].split()[-1])
      if "Test set results" in l:
        avg_acc =  100 * float( l.split()[-1] )
        avg_loss =  float( l.split()[-4] )
  return np.average(vals), np.std(vals), np.min(vals), np.max(vals), avg_loss, avg_time

for dataset in ["citeseer","cora","pubmed"]:
  print( dataset )
  for model, mname in [
      ("gcn", "GCN-3"),
      ("gcnnorm", "GCN-norm-3"),
      ("res", "RGCN-3"),
      ("resnorm", "RGCN-norm-3"),
      ("gcnfullnorm", "GCN-fullnorm"),
      ("resfullnorm", "RGCN-fullnorm"),
      ("ode", "ODE-GCN-norm-3"),
      ("odefullnorm", "ODE-GCN-fullnorm-3"),
  ]:
    try:
      acc_avg, acc_std, acc_min, acc_max, loss_avg, time_avg = get_stats(
        "{model}_{dataset}.txt".format(model=model,dataset=dataset)
      )
      print(
        "\t\t{name} & {acc_avg:.2f} & {acc_std:.2f} & {acc_min:.2f} & {acc_max:.2f} & {loss_avg:.4f} & {time_avg:.4f} \\\\".format(
          name = mname,
          acc_avg = acc_avg,
          acc_std = acc_std,
          acc_min = acc_min,
          acc_max = acc_max,
          loss_avg = loss_avg,
          time_avg = time_avg,
        )
      )
    except FileNotFoundError:
      print(
        "%\t\t{name} & {acc_avg:.2f} & {acc_std:.2f} & {acc_min:.2f} & {acc_max:.2f} & {loss_avg:.4f} & {time_avg:.4f} \\\\".format(
          name = mname,
          acc_avg = 0.,
          acc_std = 0.,
          acc_min = 0.,
          acc_max = 0.,
          loss_avg = 0.,
          time_avg = 0.,
        )
      )
  #end for
#end for
