
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


def table(args, results):
    
    print("\n")
    print(f"Epsilon: {args.epsilon}, delta: {args.delta}, k: {args.k}")

    r_unsat = []
    r_true = []
    r_false = []

    hits = 0
    for cid in results:
        rid = results[cid]
        if not rid['is_sat']:
            r_unsat.append(rid['t_total'])
        elif rid['wmi']/rid['Z'] >= args.k:
            r_true.append(rid['t_total'])
            if rid['bound_hit']:
                hits += 1
        else:
            r_false.append(rid['t_total'])

    mean_unsat = np.mean(r_unsat) if len(r_unsat) else '-'
    std_unsat = np.std(r_unsat) if len(r_unsat) else '-'
    
    mean_true = np.mean(r_true) if len(r_true) else '-'
    std_true = np.std(r_true) if len(r_true) else '-'

    mean_false = np.mean(r_false) if len(r_false) else '-'
    std_false = np.std(r_false) if len(r_false) else '-'

    print(f"ROBUST (P = 1)[{len(r_unsat)}/{len(results)}] {mean_unsat} ({std_unsat}) s.")
    print(f"ROBUST (P > {1-args.k})[{len(r_false)}/{len(results)}] {mean_false} ({std_false}) s.")
    print(f"NOT ROBUST (P <= {1-args.k})[{len(r_true)}/{len(results)}] {mean_true} ({std_true}) s.")

    print(f"Bound hits: {hits}/{len(r_true)}")
    print("\n")

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # SYS
    parser.add_argument('sys', type=str,
                        help="System")
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
                        default=16)
    parser.add_argument('--df_n_min', type=int,
                        help="DET Forest min. instances in internal nodes",
                        default=500)
    parser.add_argument('--df_n_max', type=int,
                        help="DET Forest max. instances in leaves",
                        default=1000)

    # NN arguments
    parser.add_argument('--nn_hidden', type=int, nargs='+',
                        help="NN: Hidden layer size",
                        default=[16, 16, 16])
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
    parser.add_argument('--delta', type=float,
                        help="Delta robustness in regression (def None = classification)", default=None)
    parser.add_argument('--seed', type=int, help="Seed number",
                        default=666)
    parser.add_argument('--k', type=float,
                        help="Check Pr(post|pre) >= k", default=0.5)
    parser.add_argument('--reverse_robustness', action='store_true',
                        help="Quantify LACK OF local robustness")
    parser.add_argument('--n_points', type=int,
                        help="Points for each conf. (epsilon, delta)", default=5)
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

    if args.sys not in ['df', 'nn']:
        raise NotImplementedError(f"System {sys} not implemented.")

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
    
    output_type = REAL if args.delta is not None else BOOL
    smt_y = [Symbol(vname, output_type) for vname in varnames[args.n_inputs:]]

    taskstr = 'TC' if args.delta is None else f'TR-{args.delta}'
    if args.data_folder is not None:
        data_folder = args.data_folder
    else:
        data_folder = os.path.join(f'data_N{args.n_inputs}_M{args.m}_{taskstr}/')

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

    #################### TRAINING TE

    sysstr = args.sys
    if args.sys == 'df':
        sysstr += f'_{args.df_size}'
    elif args.sys == 'nn':
        hiddenstr = '-'.join(map(str,args.nn_hidden))
        sysstr += f'_{hiddenstr}'

    data_sys = sample_data(args.n_inputs, args.sys_samples, fc, plot=True)
    data_test = sample_data(args.n_inputs, args.sys_samples//100, fc)
    
    sys_path = os.path.join(data_folder, f'{sysstr}.json')
    if os.path.isfile(sys_path):
        print(f"Found {args.sys} at: {sys_path} ...", end=" ")
        if args.sys == 'df':
            sys = DETForest.load(sys_path)
        elif args.sys == 'nn':
            sys = FFNN.load(sys_path)

    else:
        print(f"Training {args.sys.upper()}")
        if args.sys == 'df':
            sys = DETForest(feats,
                            data_sys,
                            args.df_size,
                            1,
                            n_min=args.df_n_min,
                            n_max=args.df_n_max,
                            seed=args.seed)
        elif args.sys == 'nn':
            dimensions = [args.n_inputs] + args.nn_hidden + [1]
            sys = FFNN.train_FFNN(data_sys,
                                  dimensions,
                                  args.delta is None,
                                  args.nn_epochs,
                                  batch_size=args.nn_batch_size,
                                  lr=args.nn_lr,
                                  momentum=args.nn_momentum,
                                  seed=args.seed)

        sys.save(sys_path)

    test_perf = sys.test_performance(data_test)
    if args.delta is None:
        print(f"{args.sys} accuracy: {test_perf}")
    else:
        print(f"{args.sys} MSE: {test_perf}")

    #################### INIT VERIFIER
    domain = set(smt_x)

    partition_mode, n_partitions = args.partitions.split('-')
    if args.sys == 'df':
        sys_enc = DETForestEncoder(sys, smt_x, smt_y,
                                   seed=args.seed,
                                   n_partitions=int(n_partitions),
                                   partition_mode=partition_mode
                                   )
    elif args.sys == 'nn':
        sys_enc = NNEncoder(sys, smt_x, smt_y,
                            seed=args.seed,
                            n_partitions=int(n_partitions),
                            partition_mode=partition_mode
                            )

    verifier = Verifier(sys_enc, det, domain,
                        use_ibp=args.use_ibp,
                        use_cache=True
                        )

    expstr = f'{args.epsilon}_{args.delta}_{args.k}_{args.seed}'
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

        if args.reverse_robustness:
            print("NOT", end=" ")
        if args.delta is None:
            print("(same class)")
        else:
            print(f"({args.delta}-robust)")

        key = cid
        if key in results:
            print("Result found. Skipping.")
            continue
            
        yc = fc(xc.reshape(-1, len(smt_x)))#[0]
        wmi, Z = verifier.check_local_robustness(args.k,
                                                 xc, yc,
                                                 args.epsilon,
                                                 args.delta,
                                                 reverse=args.reverse_robustness
                                                 )
        results[key] = verifier.last_run_stats

        if wmi > 0: # ignore unsat (det. robust) instances
            print(f"\n{wmi/Z} >= {args.k}: {wmi >= args.k * Z}")
            print(f"(nint: {verifier.nint_Z + verifier.nint_query}, t: {verifier.t_total} s.)")
                            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)


    table(args, results)
            
