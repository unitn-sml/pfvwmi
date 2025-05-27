
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pysmt.shortcuts import *

from pfvwmi import Verifier
from pfvwmi.encoders import DETForestEncoder, NNEncoder
from pfvwmi.models import DET, DETForest, FFNN


def sample_data(n_inputs, n_samples, fc, plot=False):

    x = np.random.normal(0, 1, size=(n_samples, n_inputs))
    y = fc(x).reshape(-1, 1)        
    xy = np.concatenate((x, y), axis=1)

    if plot:
        neg = x[y[:,0] == 0]
        pos = x[y[:,0] == 1]
        fig = plt.figure()
        nplots = n_inputs * (n_inputs - 1) // 2
        axes = fig.subplots(nplots + 1, sharex=False, sharey=False)
        fig.tight_layout()
        k = 0
        for i in range(n_inputs-1):
            for j in range(i+1, n_inputs):
                axes[k].set_xlabel(f"x{i}")
                axes[k].set_ylabel(f"x{j}")
                axes[k].scatter(neg[:,i], neg[:,j], marker='x', alpha=0.6)
                axes[k].scatter(pos[:,i], pos[:,j], marker='x', alpha=0.6)
                k += 1

        axes[k].set_xlabel(f"r1")
        axes[k].set_ylabel(f"r2")
        axes[k].scatter(neg[:,i], neg[:,j], marker='x', alpha=0.6)
        axes[k].scatter(pos[:,i], pos[:,j], marker='x', alpha=0.6)

        plt.show()
    
    return xy


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # SYS
    parser.add_argument('--sys_samples', type=int,
                        help="# training samples for SYS",
                        default=10000)
    # DET arguments
    parser.add_argument('--pop_samples', type=int,
                        help="# DET training samples",
                        default=1000)
    parser.add_argument('--pop_n_min', type=int,
                        help="DET min. instances in internal nodes",
                        default=500)
    parser.add_argument('--pop_n_max', type=int,
                        help="DET max. instances in leaves",
                        default=1000)
    # DF arguments
    parser.add_argument('--df_size', type=int,
                        help="DET Forest size",
                        default=10)
    parser.add_argument('--df_n_min', type=int,
                        help="DET Forest min. instances in internal nodes",
                        default=500)
    parser.add_argument('--df_n_max', type=int,
                        help="DET Forest max. instances in leaves",
                        default=1000)

    # NN arguments
    parser.add_argument('--nn_hidden', type=int, nargs='+',
                        help="NN: Hidden layer size",
                        default=[32, 32, 32])
    parser.add_argument('--nn_epochs', type=int,
                        help="NN: Training epochs", default=50)
    parser.add_argument('--nn_batch_size', type=int,
                        help="NN: Batch size length", default=20)
    parser.add_argument('--nn_lr', type=float,
                        help="NN: Learning rate", default=5e-3)
    parser.add_argument('--nn_momentum', type=float,
                        help="NN: Momentum", default=0.0)

    # experiment arguments
    parser.add_argument('--n_inputs', type=int,
                        help="Input dimensions",
                        default=3)
    parser.add_argument('--m', type=int,
                        help="GT mixture size",
                        default=10)
    parser.add_argument('--epsilon', type=float,
                        help="Epsilon robustness", default=0.01)
    parser.add_argument('--seed', type=int, help="Seed number",
                        default=666)
    parser.add_argument('--k', type=float,
                        help="Check Pr(post|pre) >= k", default=0.5)
    parser.add_argument('--reverse_class', action='store_true',
                        help="Quantify LACK OF local robustness")
    parser.add_argument('--n_points', type=int,
                        help="Points for each conf.", default=5)
    parser.add_argument('--partitions', type=str,
                        help="Mode / # partitions (no partitioning)",
                        default='sample-0')
    parser.add_argument('--use_ibp',
                        help="Use interval bound prop. (def. False)",
                        action='store_true')
    parser.add_argument('--data_folder', type=str,
                        help="Results folder",
                        default=None)    

    args = parser.parse_args()
    print("==================================================")
    print(args)
    print("==================================================")

    np.random.seed(args.seed)


    c1 = np.random.choice([-1, 0, 1], (args.n_inputs, args.m)) \
        * np.random.random((args.n_inputs, args.m))
    c2 = np.random.choice([-1, 1], (args.m, args.n_outputs)) \
        * np.random.random((args.m, args.n_outputs))

    def fc(x):
        y = np.matmul(np.matmul(x,c1), c2)
        if args.delta is None:
            return (y >= 0).astype(int)
        
        return y
 
    varnames = [f'x_{i}' for i in range(args.n_inputs)] + ['y']

    feats = [(vname, 'real') for vname in varnames[:args.n_inputs]]
    smt_x = [Symbol(vname, REAL) for vname, _ in feats]    
    smt_y1 = [Symbol(vname, BOOL) for vname in varnames[args.n_inputs:]]
    smt_y2 = [Symbol(f'{vname}_surrogate', BOOL) for vname in varnames[args.n_inputs:]]

    if args.data_folder is not None:
        data_folder = args.data_folder
    else:
        data_folder = os.path.join(f'data_N{args.n_inputs}_M{args.m}/')

    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    #################### TRAINING DET

    data_pop = sample_data(args.n_inputs, args.pop_samples, fc)[:,:-1]
    tv_size = int(0.9 * len(data_pop))
    train_valid = data_pop[:tv_size]
    test = data_pop[tv_size:]

    det_path = os.path.join(data_folder, 'det.json')
    if os.path.isfile(det_path):
        print(f"Found DET at: {det_path} ...", end=" ")
        det = DET.load(det_path)
    else:
        print(f"Training DET with nmin: {args.pop_n_min}, nmax: {args.pop_n_max} ...", end=" ")
        train_size = int(0.9 * len(train_valid))
        det = DET(feats,
                  train_valid[:train_size],
                  n_min=args.pop_n_min, n_max=args.pop_n_max)
        det.prune(train_valid[train_size:])
        det.save(det_path)

    test_ll = np.mean(np.log(det.evaluate(test)))
    print(f"DET size: {det.size}, test LL: {test_ll}")

    #################### TRAINING SYS 1 and SYS 2

    hiddenstr = '-'.join(map(str,args.nn_hidden))
    sysstr1 = f'nn_{hiddenstr}'    
    sysstr2 = f'df_{args.df_size}'
    sysstr = f'{sysstr1}+{sysstr2}'

    data_sys = sample_data(args.n_inputs,
                           args.sys_samples, fc, plot=True)
    data_test = sample_data(args.n_inputs,
                            args.sys_samples//100, fc)
    
    sys1_path = os.path.join(data_folder, f'{sysstr1}.json')
    if os.path.isfile(sys1_path):
        print(f"Found NN at: {sys1_path} ...", end=" ")
        sys1 = FFNN.load(sys1_path)
    else:
        print(f"Training NN")
        dimensions = [args.n_inputs] + args.nn_hidden + [1]
        sys1 = FFNN.train_FFNN(data_sys,
                               dimensions,
                               True,
                               args.nn_epochs,
                               batch_size=args.nn_batch_size,
                               lr=args.nn_lr,
                               momentum=args.nn_momentum,
                               seed=args.seed)
        sys1.save(sys1_path)

    sys2_path = os.path.join(data_folder, f'{sysstr2}.json')
    if os.path.isfile(sys2_path):
        print(f"Found DETF at: {sys2_path} ...", end=" ")
        sys2 = DETForest.load(sys2_path)
    else:
        sys2 = DETForest(feats,
                         data_sys,
                         args.df_size,
                         1,
                         n_min=args.df_n_min,
                         n_max=args.df_n_max,
                         seed=args.seed)
        sys2.save(sys2_path)

    test_perf1 = sys1.test_performance(data_test)
    print(f"NN accuracy: {test_perf1}")
    test_perf2 = sys2.test_performance(data_test)
    print(f"DETF accuracy: {test_perf2}")

    #################### INIT VERIFIER
    domain = set(smt_x)

    partition_mode, n_partitions = args.partitions.split('-')
    sys1_enc = NNEncoder(sys1, smt_x, smt_y1,
                         seed=args.seed,
                         n_partitions=int(n_partitions),
                         partition_mode=partition_mode)

    sys2_enc = DETForestEncoder(sys2, smt_x, smt_y2, seed=args.seed)
    verifier = Verifier(sys1_enc, det, domain,
                        use_ibp=args.use_ibp,
                        use_cache=True)

    expstr = f'{args.epsilon}_{args.k}_{args.seed}'
    expstr += f'_{args.partitions}_' + ('ibp' if args.use_ibp else 'noibp')
    results_path = os.path.join(data_folder, f'results_{sysstr}_{expstr}.json')
    if os.path.isfile(results_path):
        print("Found results at:", results_path)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

    else:
        results = dict()

    for cid in range(args.n_points):
        xc = np.array(data_pop[cid])
        print("==================================================")
        print(f"[{cid+1}/{args.n_points}]({args.epsilon}-ball around point: {xc}) =>", end=" ")

        if args.reverse_class:
            print("NOT", end=" ")
        print("(y1 <=> y2)")

        key = cid
        if key in results:
            print("Result found. Skipping.")
            continue

        wmi, Z = verifier.check_local_equivalence(sys2_enc,
                                                  args.k,
                                                  xc,
                                                  args.epsilon,
                                                  reverse=args.reverse_class)

        results[key] = verifier.last_run_stats

        if wmi > 0: # ignore unsat (det. robust) instances
            print(f"\n{wmi/Z} >= {args.k}: {wmi >= args.k * Z}")
            print(f"(nint: {verifier.nint_Z + verifier.nint_query}, t: {verifier.t_total} s.)")
                            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

            
