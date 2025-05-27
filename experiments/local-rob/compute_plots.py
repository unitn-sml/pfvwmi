
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sys import argv

plt.rcParams["text.usetex"] =True

K = 0.1

# plot params
LOG = False
CUMULATIVE = False
LINEWIDTH = 2.0
LINEALPHA = 0.8
EXTENSION='pdf'


if len(argv) > 1:
    data_folder = argv[1]
else:
    data_folder = os.path.join(f'JAIR/')


partitioning = {
    'sample-0' : 'No part.',
    'sample-16' : 'Part. (sampling)',
}


epsilons = [0.1, 0.15, 0.2]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title(f'NN $\\epsilon$-robustness verification')
ax.set_xlabel(f'Instances')
if CUMULATIVE:
    ax.set_ylabel('Cumulative runtime (s)')
else:
    ax.set_ylabel('Runtime (s)')
if LOG:
    ax.set_yscale("log")
        
for part in partitioning:
    r_unsat = []
    r_true = []
    r_false = []
    hits = 0
    for epsilon in epsilons:

        expstr = f'{epsilon}_None_{K}_666'
        expstr += f'_{part}_ibp'
        resstr = f'results_nn_32-32-32_{expstr}.json'
        results_path = os.path.join(data_folder, resstr)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        for cid in results:
            rid = results[cid]
            if not rid['is_sat']:
                r_unsat.append(rid['t_total'])
            elif rid['wmi']/rid['Z'] >= K:
                r_true.append(rid['t_total'])
                if rid['bound_hit']:
                    hits += 1
            else:
                r_false.append(rid['t_total'])


    y = sorted(r_true + r_false)
    if CUMULATIVE: y = [sum(y[:i]) for i in range(len(y))]
    x = list(range(len(y)))
    print("LEN:", len(y))
    label = partitioning[part]
    ax.plot(x, y,
            label=label,
            linestyle="-" if part == "sample-0" else "--",
            linewidth=LINEWIDTH,
            alpha=LINEALPHA)

ax.legend(loc='upper left')
iscum = '_cumulative' if CUMULATIVE else ''
plt.savefig(f"plot_rob{iscum}_NN.{EXTENSION}", bbox_inches="tight")
plt.show()
plt.close()
