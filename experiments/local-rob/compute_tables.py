
import numpy as np
import os
import pickle

from sys import argv

if len(argv) > 1:
    data_folder = argv[1]
else:
    data_folder = os.path.join(f'JAIR/')

rowstr = {
    ("noibp","sample-0") : "No BP",
    ("ibp","sample-0") : "BP"
}

epsilons = [0.01, 0.05, 0.1]
header = "\\begin{tabular}{|c|"
header += "c|" * len(epsilons)
header += "}\n  \\hline\n"

for epsilon in epsilons:
    header += " & $\\epsilon = " + str(epsilon) + "$"

nopart_table = [header + " \\\\ \\hline\\hline \n"]

nopart_table.append(" \\multicolumn{4}{|c|}{NN} \\\\ \\hline")    
for ibp in ['noibp', 'ibp']:
    for part in ['sample-0']:

        runtime_row = f" {rowstr[(ibp,part)]}"

        for epsilon in epsilons:
            expstr = f'{epsilon}_None_0.1_666'
            expstr += f'_{part}_{ibp}'
            resstr = f'results_nn_32-32-32_{expstr}.json'
            results_path = os.path.join(data_folder, resstr)
            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            r_aggr = []            
            for cid in results:
                rid = results[cid]
                r_aggr.append(rid['t_total'])                

            mean_aggr = f"{np.mean(r_aggr):.2f}" if len(r_aggr) else '-'
            std_aggr = f"{np.std(r_aggr):.2f}" if len(r_aggr) else '-'

            runtime_row += " & \\mstd{" + mean_aggr + "}{" + std_aggr + "}"

        nopart_table.append(runtime_row + " \\\\")

nopart_table.append("\\hline\\hline\n")

nopart_table.append("\\end{tabular}\n")
print("\n".join(nopart_table))
with open(f'tab_localrob.tex', 'w') as f:
    f.write("\n".join(nopart_table))
