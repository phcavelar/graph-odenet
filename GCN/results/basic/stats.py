import sys
import numpy as np
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt

dataset = sys.argv[1]
models = [m for m in sys.argv[2:]]

model_name = {
  "gcn": "GCN-3",
  "gcnnorm": "GCN-norm-3",
  "res": "RGCN-3",
  "resnorm": "RGCN-norm-3",
  "gcnfullnorm": "GCN-fullnorm",
  "resfullnorm": "RGCN-fullnorm",
  "ode": "ODE-GCN-norm-3",
  "odefullnorm": "ODE-GCN-fullnorm-3",
}

model_color = {
  "gcn": "#6457a6",
  "gcnnorm": "#6457a6",
  "res": "#664e4c",
  "resnorm": "#664e4c",
  "gcnfullnorm": "#6457a6",
  "resfullnorm": "#664e4c",
  "ode": "#9b8816",
  "odefullnorm": "#9b8816",
}

model_vals = []
for m in models:
  vals = []
  with open("{}_{}.txt".format(m,dataset)) as f:
    for l in f:
      if "Run" in l:
        vals.append(float(l[-7:-2]))
  model_vals.append( np.array(vals) )

colors = ["#6457a6","#664e4c","#9b8816","#8c271e","#002400","#000000",]

plt.hist([v for v in model_vals], bins=25, normed=True, color=[model_color[m] for m in models],label=[model_name[m] for m in models])
plt.legend()
plt.xlabel("Accuracy (%)")
plt.ylabel("Bins")
plt.savefig("hist_{}_{}.pdf".format(dataset,"__".join(models)), bbox_inches="tight")

for m, v in zip(models,model_vals):
  print("Shapiro {m}".format(m=m), sps.shapiro(v))
  

for i in range(len(models)):
  for j in range(i+1,len(models)):
    m1, v1 = models[i], model_vals[i]
    m2, v2 = models[j], model_vals[j]
    if m1!=m2:
      print("{m1} x {m2}".format(m1=m1, m2=m2))
      print("mannwhitneyu", sps.mannwhitneyu(v1,v2))
      print("kruskal", sps.kruskal(v1,v2))
