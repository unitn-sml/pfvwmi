
from itertools import product
import numpy as np
from pysmt.shortcuts import *
import torch

from pfvwmi.utils import Itex

from .encoder import Encoder, Split

class DETForestEncoder(Encoder):

    def __init__(self, df, X, Y,
                 smt_env=None,
                 seed=None,
                 n_partitions=8,
                 partition_mode='sample'
                 ):
        super().__init__(df, X, Y, smt_env,
                         seed=seed,
                         n_partitions=n_partitions,
                         partition_mode=partition_mode)
        assert(self.model.discrete_output)


    def smt_formula(self, bounds=None):
        self.splits = []
        clauses = []
        for j, yj in enumerate(self.Y):
            vars_yj = [[],[]]
            for i, dets in enumerate(self.model.forest[j]):
                for b, var in enumerate([Symbol(f'p_y{j}_neg_{i}', REAL),
                                         Symbol(f'p_y{j}_pos_{i}', REAL)]):

                    vars_yj[b].append(var)
                    enc = dets[b].smt_weight(formula_var=var, bounds=bounds)
                    clauses.append(enc)
                    t = dets[b].root
                    if self.partition_mode.startswith('all'):
                        self._add_all_splits(t)
                    elif not t.is_leaf:
                        assert(t.split_val is not None)
                        atom = LE(self.X[t.split_var],
                                  Real(float(t.split_val)))
                        test = lambda x, t=t : x[:,t.split_var] <= t.split_val
                        self.splits.append(Split(atom, test))

            clauses.append(Iff(yj, LE(Plus(*vars_yj[0]), Plus(*vars_yj[1]))))

        return And(*clauses)

    def _add_all_splits(self, t):
        if not t.is_leaf:
            assert(t.split_val is not None)
            atom = LE(self.X[t.split_var],
                      Real(float(t.split_val)))
            test = lambda x, t=t : x[:,t.split_var] <= t.split_val
            self.splits.append(Split(atom, test))
            self._add_all_splits(t.children[0])
            self._add_all_splits(t.children[1])
        
