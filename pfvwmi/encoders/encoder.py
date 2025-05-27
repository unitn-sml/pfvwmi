
from itertools import product
import numpy as np
from pysmt.shortcuts import *

class Split:

    def __init__(self, atom, test):
        self.atom = atom
        self.test = test

    def eval(self, samples, px, Z):
        ind = np.array(self.test(samples), dtype=int).reshape((-1, 1))
        assert(ind.shape == (len(samples),1)), ind.shape
        assert(len(np.unique(ind)) <= 2), np.unique(ind)
        p_split_post = np.sum(ind * px.reshape(-1,1)) / Z
        assert(0 <= p_split_post and p_split_post <= 1), p_split_post
        return p_split_post


class Encoder:

    PMODES = {'sample', 'allsample', 'random', 'allrandom'}

    def __init__(self, sys, X, Y,
                 smt_env=None,
                 seed=None,
                 n_partitions=8,
                 partition_mode='sample'
                 ):

        ''' Default constructor.
        sys (Encoder) - the system
        X (list(FNode)) - pysmt input variables
        Y (list(FNode)) - pysmt output variables
        seed (int) - seed number
        use_cache (bool) - WMI with caching
        use_ibp (bool) - use interval bound propagation during encodings
        tighten_bounds (bool) - attempt at tightening the input bounds
        n_partitions (int) - number of partitions
        partition_mode (str) - how to partition

        '''
        assert(partition_mode in Encoder.PMODES)
        self.model = sys
        self.X = X
        self.Y = Y
        self.smt_env = get_env() if smt_env is None else smt_env
        self.seed = seed
        self.splits = []
        self.n_partitions = n_partitions
        self.partition_mode = partition_mode


    def smt_formula(self, bounds=None):
        raise NotImplementedError("Implement this!")

    def partition(self, pre, post, bounds, w=None):

        if self.n_partitions == 0 or len(self.splits) == 0:
            # got nothing to partition
            return [Bool(True)]
        
        k = np.log2(self.n_partitions).astype(int)

        if 'sample' in self.partition_mode:
            samples = self.draw_samples(pre, post, bounds)
        elif 'random' in self.partition_mode:
            samples = []
        else:
            raise NotImplementedError(f"Partition mode not implemented: {self.partition_mode}")

        if len(samples) == 0:
            s_atoms = [s.atom
                       for s in np.random.choice(self.splits,
                                                 min(k, len(self.splits)),
                                                 replace=False)]
        else:
            if w is None: w = lambda x : np.ones((len(x), 1))
            px = w(samples)
            Z = np.sum(px)

            candidates = [(s.atom, s.eval(samples, px, Z))
                          for s in self.splits]
            top_k = sorted(candidates, key=lambda x: abs(x[1]-1/2))[:k]

            s_atoms = []
            print(f"Top (k = {k}) splits:")
            for s_atom, p_s in top_k:
                print(f"P({s_atom} | pre & post) : {p_s}")
                s_atoms.append(s_atom)
                
        # construct partition clauses
        partitions = []
        #s_atoms.reverse()
        k = min(k, len(s_atoms))
        for swap in product([False, True], repeat=k):
            clause = []
            for i in range(k):
                clause.append(Not(s_atoms[i]) if swap[i] else s_atoms[i])

            partitions.append(And(*clause))

        return partitions


    def draw_samples(self, pre, post, bounds, min_samples=1e3, max_tries=3):
        '''Returns a uniform sample that satisfy:
        
            (pre(x) & post(sys(x)))

        Tries to obtain a 'min_samples' with at most 'max_tries'
        rounds of sampling from 'bounds'.

        '''
        flter_x = lambda xi : is_sat(pre.substitute(
            {v : Real(float(xi[j])) for j, v in enumerate(self.X)}))
        if self.model.discrete_output:
            flter_y = lambda yi : is_sat(post.substitute(
                {v : Bool(bool(yi[j])) for j, v in enumerate(self.Y)}))
        else:
            flter_y = lambda yi : is_sat(post.substitute(
                {v : Real(float(yi[j])) for j, v in enumerate(self.Y)}))

        samples = np.array([])
        it = 0
        print("Sampling condition statistics:")
        while len(samples) < min_samples and it < max_tries:
            it += 1
            n_samples = min_samples - len(samples) if samples is not None else min_samples
            x_norm = np.random.random(int(min_samples) *
                                      len(self.X)).reshape(-1, len(self.X))
            b = np.array(bounds).T
            x_scaled = b[0] + (b[1]-b[0]) * x_norm
            x_filtered = np.array([v for v in x_scaled
                                   if flter_x(v)
                                   and flter_y(self.model.evaluate(v.reshape(1, -1)).flatten())]).reshape((-1, len(self.X)))

            if len(samples) == 0:
                samples = x_filtered
            else:
                samples = np.concatenate((samples, x_filtered))

            print(f" it: {it}/{max_tries} -- sampled: {len(samples)}/{min_samples} (success ratio: {len(x_filtered)/len(x_scaled)})")

        return samples
