
import numpy as np
import os
import pickle
import torch
from pysmt.shortcuts import *
from time import time

try:
    from wmipa import WMISolver

except ModuleNotFoundError:
    WMISolver = None
from pfvwmi.encoders import NNEncoder
from pfvwmi.models.dataset import Dataset
from pfvwmi.models.det import DET
from pfvwmi.models.neuralnet import FFNN
from pfvwmi.io import Density
from pfvwmi.utils import clone_var, monotonicity,\
    self_compose_formula, self_compose_weight

CACHE_MODE = 0


def train_dets(pop_model_path, pop_data, n_min, n_max,
               test_split=0.666, valid_split=0.1):    
    """Trains 2 distinct DETs for male/female individuals."""

    # train / test split
    train_size = int((1 - test_split) * pop_data.size[0])
    pop_train = pop_data.values[:train_size]
    pop_test = pop_data.values[train_size:]
    m_test = pop_test[pop_test[:,0] == 0][:,1:]
    f_test = pop_test[pop_test[:,0] == 1][:,1:]

    if os.path.isfile(pop_model_path):
        print("Found DETS at:", pop_model_path)
        with open(pop_model_path, 'rb') as f:
            det_m, det_f = pickle.load(f)

    else:
        # drop the 'gender' feature
        pop_feats = [(v.symbol_name(), 'real')
                     for v in pop_data.smt_variables()[1:]]

        # split the training data according to the 'gender' value
        m_train = pop_train[pop_train[:,0] == 0][:,1:]        
        f_train = pop_train[pop_train[:,0] == 1][:,1:]

        m_size = int(len(m_train) * (1 - valid_split))
        f_size = int(len(f_train) * (1 - valid_split))

        print("Training dual population model (DET)")
        print(f"Hyperparams: NMIN = {n_min}, NMAX = {n_max}")
        print(f"Data: |train| = {len(pop_train)}")
        
        det_m = DET(pop_feats, m_train[:m_size],
                    n_min=n_min, n_max=n_max)

        det_m.prune(m_train[m_size:])

        det_f = DET(pop_feats, f_train[:f_size],
                    n_min=n_min, n_max=n_max)
    
        det_f.prune(f_train[f_size:])

        with open(pop_model_path, 'wb') as f:
            pickle.dump((det_m, det_f), f)        
    
    test_ll = np.mean(np.log(det_m.evaluate(m_test)))
    test_ll += np.mean(np.log(det_f.evaluate(f_test)))
    test_ll = test_ll / 2
    print(f"DET test (mean) log-likelihood: {test_ll}")
    print(f"DET size: {det_m.size + det_f.size + 1}\n")

    return det_m, det_f


def train_nets(pred_model_prefix, train_dataset, args):

    model_path_f = lambda i : pred_model_prefix + f'_{i}'

    checkpoint_delta = args.epochs//args.n_checkpoints
    checkpoint_interval = list(range(0, args.epochs+1,
                                     checkpoint_delta))

    print("CHECKPOINTS:", checkpoint_interval)
    
    paths = [model_path_f(i) for i in checkpoint_interval]

    if not all([os.path.isfile(p) for p in paths]):
        
        print("Training predictive model (FFNN)")
        print(f"Hyperparams: HIDDEN = {args.hidden}")
        print(f"Data: |train| = {train_dataset.size[0]}")

        dimensions = [train_dataset.size[1] - 1] + args.hidden + [1]
        discrete_output = True
        _ = FFNN.train_FFNN(train_dataset.values,
                            dimensions,
                            discrete_output,
                            args.epochs,
                            model_path_f=model_path_f,
                            checkpoint_delta=checkpoint_delta,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            momentum=args.momentum,
                            seed=args.seed)

    nets = []
    for i, p in enumerate(paths):
        print(f"Loading predictive model (FFNN) from: '{p}'")
        nets.append((checkpoint_interval[i], FFNN.load(p)))

    return nets



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help="Unbiased/biased")
    
    # DET arguments
    parser.add_argument('--n_min', type=int, help="DET: Min. instances in internal nodes",
                        default=1000)
    parser.add_argument('--n_max', type=int, help="DET: Max. instances in leaves",
                        default=2000)

    # NN arguments
    parser.add_argument('--hidden', type=int, nargs='+', help="NN: Hidden layer size",
                        default=[8, 8])
    parser.add_argument('--epochs', type=int, help="NN: Training epochs", default=1000)
    parser.add_argument('--batch_size', type=int, help="NN: Batch size length", default=200)
    parser.add_argument('--lr', type=float, help="NN: Learning rate", default=5e-3)
    parser.add_argument('--momentum', type=float, help="NN: Momentum", default=0.0)
    parser.add_argument('--seed', type=int, help="Seed number", default=666)

    # experiment arguments
    parser.add_argument('--n_checkpoints', type=int, help="N.checkpoints", default=10)
    parser.add_argument('--no-exec', help="Generates benchmarks only", action='store_true')
    parser.add_argument('--no-pop', help="Ignore population model", action='store_true')
    parser.add_argument('--encoder', type=int, help="NN encoder", default=0)
    #parser.add_argument('--approx', help="Approximate integration", action='store_true')

    args = parser.parse_args()

    print("==================================================")
    print(args)
    print("==================================================")

    if args.exp not in ['unbiased', 'biased']:
        print("exp should be 'unbiased' or 'biased")
        exit(1)

    data_folder = f'data_{args.exp}/'
    pop_dataset = Dataset.load(os.path.join(data_folder, 'pop_data.json'))
    pred_dataset = Dataset.load(os.path.join(data_folder, 'pred_data.json'))    
    train_dataset, test_dataset = pred_dataset.split(0.9)

    out_folder = os.path.join(f'out_{args.exp}/')
    pop_str = 'nopop' if args.no_pop else 'pop'
    enc_str = f'enc{args.encoder}'
    benchmark_folder = os.path.join(out_folder, f'benchmarks_{pop_str}_{enc_str}/')

    pop_model_path = os.path.join(out_folder, 'population.model')
    pred_model_path = os.path.join(out_folder, 'predictive.model')

    results_prefix = f'results_{pop_str}'

    results_file = os.path.join(out_folder, f'{results_prefix}.json')

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    if not os.path.isdir(benchmark_folder):
        os.mkdir(benchmark_folder)

    # retrieving the smt variables
    female, hpw, yexp, hw = pop_dataset.smt_variables()
    y = pred_dataset.smt_variables()[-1]
    net_input = [hpw, yexp, hw]
    net_output = [y]

    if not args.no_pop:
        det_m, det_f = train_dets(pop_model_path, pop_dataset,
                                  args.n_min, args.n_max)

        # encoding the population model
        f_pop_m = det_m.smt_formula()
        f_pop_f = det_f.smt_formula()

        f_pop = And(Implies(Not(female), f_pop_m),
                    Implies(female, f_pop_f))

        w_pop_m = det_m.smt_weight()
        w_pop_f = det_f.smt_weight()
        w_pop = Times(Real(0.5), Ite(female, w_pop_f, w_pop_m))

        # bounds for more efficient NN encodings
        boundprop = []
        for n_feat in range(len(net_input)):
            lb = min(det_m.root.bounds[0][0][n_feat],
                     det_f.root.bounds[0][0][n_feat])
            ub = max(det_m.root.bounds[0][1][n_feat],
                     det_f.root.bounds[0][1][n_feat])
            boundprop.append([lb, ub])

    else:
        clauses = []
        boundprop = []
        for n_feat, feat in enumerate(net_input):
            lb = float(np.min(pop_dataset.values[:, n_feat+1]))
            ub = float(np.max(pop_dataset.values[:, n_feat+1]))
            clauses.extend([LE(Real(lb), feat),
                            LE(feat, Real(ub))])
            boundprop.append([lb, ub])

        f_pop = And(*clauses, Or(female, Not(female)))
        
        # computing the uninformed (uniform) prior
        solver = WMISolver(f_pop, Real(1))
        vol, _ = solver.computeWMI(Bool(True), net_input,
                                   cache=CACHE_MODE)
        w_pop = Real(float(1/vol))

    #solver = WMISolver(f_pop, w_pop)
    #print(f"DEBUG: Computing Z")
    #Z, nint = solver.computeWMI(Bool(True), cache=CACHE_MODE)        
    #print(f"DEBUG: Done. WMI: {Z} (should be 1.0), #int: {nint}")
    #print(f"DEBUG: Computing p(female)")
    #p_female, nint = solver.computeWMI(female, cache=CACHE_MODE)
    #print(f"DEBUG: Done. WMI: {p_female} (should be .5), #int: {nint}")

    # train neural nets
    nets = train_nets(pred_model_path, train_dataset, args)

    ########## MONOTONICITY
    # all postconditions here are the same formula
    mono_hpw_pre, mono_post = monotonicity(hpw, {yexp, hw}, y)
    mono_yexp_pre, _ = monotonicity(yexp, {hpw, hw}, y)
    mono_hw_pre, _ = monotonicity(hw, {hpw, yexp}, y)

    ########## ROBUSTNESS
    max_noise = 15
    noise = Symbol('noise', REAL)
    f_noise = And(LE(Real(0), noise),
                  LE(noise, Real(max_noise)))
    w_noise = Plus(Times(Real(-(2/max_noise) / max_noise), noise),
                   Real(2/max_noise))
            
    rob_post = Iff(y, clone_var(y))
    rob_pre = And(Equals(hpw, clone_var(hpw)),
                  Equals(hw, clone_var(hw)),
                  Equals(yexp, Plus(clone_var(yexp), noise)))

    results = []
    for n_cp, epoch_net in enumerate(nets):

        ########## empirical accuracy

        epoch, net = epoch_net
        print()
        print(f"--- Eval Epoch: {epoch} [{n_cp+1}/{len(nets)}]")
        acc = net.test_performance(test_dataset.values)
        print("Accuracy:", acc)

        print("Storing the problem encodings.")

        ########## base encoding and checks

        if args.encoder <= 1:
            boundprop = None

        enc = NNEncoder(net, net_input, net_output, seed=args.seed)

        f_pred = enc.smt_formula(
            redundancies=(args.encoder > 0),
            bounds=boundprop)

        domain_std = {v : None for v in [hpw, yexp, hw]}

        if not is_sat(And(f_pop, f_pred)):
            print("DET + NN encoding is UNSAT, aborting.")
            exit(1)

        ########## demographic parity

        f_parity = f_pop
        w_parity = w_pop
        domain_parity = domain_std
        queries_parity = [
            female,
            And(y, f_pred, female),
            And(y, f_pred, Not(female))
        ]

        path_parity = os.path.join(benchmark_folder, f'benchmark_parity_{n_cp}.json')
        Density(f_parity, w_parity, domain_parity,
                queries_parity).to_file(path_parity)

        ########## monotonicity (wrt population model)

        f_mono = self_compose_formula(f_pop)
        w_mono = self_compose_weight(w_pop)
        domain_mono = dict(domain_std)
        domain_mono.update({clone_var(v) : None for v in domain_std})

        queries_mono = [
            mono_yexp_pre, And(mono_yexp_pre, mono_post, self_compose_formula(f_pred)),
            mono_hw_pre, And(mono_hw_pre, mono_post, self_compose_formula(f_pred)),
            mono_hpw_pre, And(mono_hpw_pre, mono_post, self_compose_formula(f_pred))
        ]

        path_mono = os.path.join(benchmark_folder, f'benchmark_mono_{n_cp}.json')
        Density(f_mono, w_mono, domain_mono,
                queries_mono).to_file(path_mono)

        ########## robustness to yexp noise

        f_rob = And(f_pop, f_noise)
        w_rob = Times(w_pop,  w_noise)
        domain_rob = dict(domain_std)
        domain_rob.update({clone_var(v) : None for v in domain_std})
        domain_rob[noise] = None
        queries_rob = [
            rob_pre,
            And(rob_pre, rob_post, self_compose_formula(f_pred))
        ]

        path_rob = os.path.join(benchmark_folder, f'benchmark_rob_{n_cp}.json')
        Density(f_rob, w_rob, domain_rob,
                queries_rob).to_file(path_rob)


        if not args.no_exec and WMISolver is not None:
            
            solver = WMISolver(f_parity, w_parity)
            wmi, parity_t, parity_nint = [], 0, 0
            for q in queries_parity:
                t0 = time()
                res = solver.computeWMI(q, domain_parity,
                                        cache=CACHE_MODE)
                parity_t += (time() - t0)
                parity_nint += res[1]
                wmi.append(res[0])

            try:
                parity = (wmi[1]/wmi[0])/(wmi[2]/(1-wmi[0]))
            except ZeroDivisionError:
                print("Division by zero when computing demographic parity, aborting.")
                print("f", wmi[0], "fp", wmi[1], "mp", wmi[2])
                exit(1)

            print("Demographic parity:", parity)
            print("runtime:", parity_t, "# integrals:", parity_nint)

            solver = WMISolver(f_mono, w_mono)
            wmi, times, nints = [], [], []
            for q in queries_mono:
                t0 = time()
                res = solver.computeWMI(q, domain_mono,
                                        cache=CACHE_MODE)
                times.append(time() - t0)
                nints.append(res[1])
                wmi.append(res[0])

            mono_yexp_t = times[0] + times[1]
            mono_yexp_nint = nints[0] + nints[1]
            mono_yexp = wmi[1] / wmi[0]
            print("Monotonicity(yexp):", mono_yexp)
            print("runtime:", mono_yexp_t, "# integrals:", mono_yexp_nint)

            mono_hw_t = times[2] + times[3]
            mono_hw_nint = nints[2] + nints[3]
            mono_hw = wmi[3] / wmi[2]
            print("Monotonicity(hw):", mono_hw)
            print("runtime:", mono_hw_t, "# integrals:", mono_hw_nint)

            mono_hpw_t = times[4] + times[5]
            mono_hpw_nint = nints[4] + nints[5]
            mono_hpw = wmi[5] / wmi[4]
            print("Monotonicity(hpw):", mono_hpw)
            print("runtime:", mono_hpw_t, "# integrals:", mono_hpw_nint)

            solver = WMISolver(f_rob, w_rob)
            wmi, rob_yexp_t, rob_yexp_nint = [], 0, 0
            for q in queries_rob:
                t0 = time()
                res = solver.computeWMI(q, domain_rob,
                                        cache=CACHE_MODE)
                rob_yexp_t += (time() - t0)
                rob_yexp_nint += res[1]
                wmi.append(res[0])

            rob_yexp = wmi[1]/wmi[0]
            print("Robustness(yexp):", rob_yexp)
            print("runtime:", rob_yexp_t, "# integrals:", rob_yexp_nint)

            res = {'epoch' : epoch,
                   'acc' : acc,

                   'parity' : parity,
                   'parity_t' : parity_t,
                   'parity_nint' : parity_nint,

                   'rob_yexp' : rob_yexp,
                   'rob_yexp_t' : rob_yexp_t,
                   'rob_yexp_nint' : rob_yexp_nint,

                   'mono_yexp' : mono_yexp,
                   'mono_yexp_t' : mono_yexp_t,
                   'mono_yexp_nint' : mono_yexp_nint,

                   'mono_hw' : mono_hw,
                   'mono_hw_t' : mono_hw_t,
                   'mono_hw_nint' : mono_hw_nint,
                   
                   'mono_hpw' : mono_hpw,
                   'mono_hpw_t' : mono_hpw_t,
                   'mono_hpw_nint' : mono_hpw_nint}

            results.append(res)

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

                
                


    

    
    



