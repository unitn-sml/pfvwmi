
from itertools import product
import numpy as np
from pysmt.shortcuts import *
import torch

from pfvwmi.utils import Itex

from .encoder import Encoder, Split

class NNEncoder(Encoder):

    def __init__(self, net, X, Y,
                 smt_env=None,
                 seed=None,
                 n_partitions=8,
                 partition_mode='sample'
                 ):

        super().__init__(net, X, Y, smt_env,
                         seed=seed,
                         n_partitions=n_partitions,
                         partition_mode=partition_mode)
        assert(len(X) == net.dimensions[0])
        assert(len(Y) == net.dimensions[-1])

        ################################################## VARIABLE DEFs

        self.common_clauses = []
        self.X_real = []
        smtvar = lambda x, y : self.smt_env.formula_manager.get_or_create_symbol(x, y)
        for var in self.X:
            if var.symbol_type() == REAL:
                self.X_real.append(var)
            else:
                auxvar = smtvar(f'{var.symbol_name()}_aux', REAL)
                auxclause = Itex(var, Equals(auxvar, Real(1)),
                                      Equals(auxvar, Real(0)))

                self.X_real.append(auxvar)
                self.common_clauses.append(auxclause)

        if self.model.discrete_output:
            if len(Y) == 1:
                var = Y[0]
                auxvar = smtvar(f'{var.symbol_name()}_aux', REAL)
                auxclause = Ite(GE(auxvar, Real(0)), var, Not(var))
                self.Y_real = [auxvar]
                self.common_clauses.append(auxclause)

            else:
                self.Y_real = [smtvar(f'{var.symbol_name()}_aux', REAL)
                     for var in Y]

                for i in range(len(self.Y_real)):
                    igreater = And([LE(self.Y_real[j], self.Y_real[i])
                                    for j in range(len(self.Y_real)) if j != i])
                    
                    itrue = And([var if j == i else Not(var)
                                 for j,var in enumerate(Y)])

                    self.common_clauses.append(Implies(igreater, itrue))

        else:
            self.Y_real = self.Y
            
        self.L_vars, self.H_vars = [], []
        for i, h_dim in enumerate(self.model.dimensions[1:-1]):
            # linear combination and output of the j-th neuron in the i-th layer
            self.L_vars.append([smtvar(f'l{i}_{j}', REAL) for j in range(h_dim)])
            self.H_vars.append([smtvar(f'h{i}_{j}', REAL) for j in range(h_dim)])

        params = list(map(lambda m : m.cpu().detach().numpy(),
                          self.model.parameters()))

        self.W, self.b = [], []
        for i in range(0, len(params), 2):
            self.W.append(params[i])
            self.b.append(params[i+1])


    def relu_enc(self, layer, index):
        condition = LE(Real(0), self.L_vars[layer][index])
        does = Equals(self.H_vars[layer][index], self.L_vars[layer][index])
        doesnt = Equals(self.H_vars[layer][index], Real(0))
        return condition, does, doesnt

    def redundant_enc(self, layer, index):
        red1 = LE(Real(0), self.H_vars[layer][index])
        red2 = LE(self.L_vars[layer][index], self.H_vars[layer][index])


    def smt_formula(self, redundancies=False, bounds=None):

        if bounds is not None:
            bounds = {(0, i) : bounds[i]
                         for i in range(self.model.dimensions[0])}

        n_tot, n_rem, n_sim = 0, 0, 0
        self.splits = []
        clauses = list(self.common_clauses)
        for k in range(0, len(self.model.dimensions)-2): # hidden layers

            # k-th layer
            in_dim_k = self.model.dimensions[k]
            out_dim_k = self.model.dimensions[k+1]
            inputs_k = self.X_real if k == 0 else self.H_vars[k-1]
            
            for o in range(out_dim_k):

                n_tot += 1
                lcomb = Plus(Real(float(self.b[k][o])),
                             *[Times(Real(float(self.W[k][o,i])), inputs_k[i])
                               for i in range(in_dim_k)
                               if (bounds is None) or bounds[(k,i)] is not None])

                # linear + relu
                linear = Equals(self.L_vars[k][o], lcomb)
                cond, cond_true, cond_false = self.relu_enc(k, o)
                activation = Itex(cond, cond_true, cond_false)

                if bounds is not None:
                    # computing bounds to the linear combination
                    lin_low = sum(self.W[k][o,i] * bounds[(k,i)][int(self.W[k][o,i] < 0)]
                                  for i in range(in_dim_k)
                                  if bounds[(k,i)] is not None) + self.b[k][o]
                    lin_up = sum(self.W[k][o,i] * bounds[(k,i)][int(self.W[k][o,i] >= 0)]
                                 for i in range(in_dim_k)
                                 if bounds[(k,i)] is not None) + self.b[k][o]

                    if lin_up <= 0: # Neuron (k,o) never activates/fires
                        bounds[(k+1,o)] = None
                        n_rem += 1
                        continue # the neuron is not encoded

                    bounds[(k+1,o)] = (lin_low, lin_up)
                    if lin_low >= 0: # Neuron (k,o) always activates/fires
                        bypass_activation = Equals(self.H_vars[k][o], lcomb)
                        clauses.append(bypass_activation)
                        n_sim += 1
                        continue

                clauses.extend([linear, activation])
                if redundancies:
                    clauses.extend(self.redundant_enc(k, o))

                if k == 0 or self.partition_mode.startswith('all'):
                    atom = cond
                    #def test(x, o=o):
                    #    return np.matmul(x, self.W[0][o,:].reshape((-1, 1))) + self.b[0][o] > 0
                    test = lambda x,o=o : np.matmul(x, self.W[0][o,:].reshape((-1, 1))) + self.b[0][o] > 0

                    self.splits.append(Split(atom, test))
                    

        for o in range(self.model.dimensions[-1]): # output layer

            lcomb = Plus(Real(float(self.b[-1][o])),
                           *[Times(Real(float(self.W[-1][o,i])), self.H_vars[-1][i])
                             for i in range(self.model.dimensions[-2])
                             if bounds is None or
                             bounds[(len(self.model.dimensions)-2, i)] is not None])
            # linear only
            linear = Equals(self.Y_real[o], lcomb)
            clauses.append(linear)

        if bounds is not None:
            print(f"Neurons: {n_tot - n_rem - n_sim}" +
                  f" unstable. {n_rem} removed. {n_sim} simplified.")

        return And(*clauses)
