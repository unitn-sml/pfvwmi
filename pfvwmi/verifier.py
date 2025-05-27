
from math import prod
from pysmt.shortcuts import *
from pysmt.optimization.goal import MaximizationGoal, MinimizationGoal

from time import time
import torch

from wmipa import WMISolver

from pfvwmi.utils import *


class Verifier:

    '''Implements a baseline WMI-based  verifier'''

    def __init__(self, sys_encoder, prior, domain,
                 use_cache=True,
                 use_ibp=True
                 ):

        ''' Default constructor.
        sys_encoder (Encoder) - encoder wrapper around sys
        prior (Prior) - the prior
        domain (set(FNode)) - integration domain
        use_cache (bool) - WMI with caching
        use_ibp (bool) - use interval bound propagation during encodings

        '''
        self.sys_encoder = sys_encoder
        self.prior = prior
        self.loose_bounds = prior.get_bounds([x.symbol_name()
                                              for x in sys_encoder.X])
        self.domain = domain
        self.cache = 0 if use_cache else -1
        self.use_ibp = use_ibp
        self.solver = None

    @property
    def last_run_stats(self):
        return {
            'is_sat' : self.is_sat,
            
            'bound_hit' : self.bound_hit,

            'Z' : self.Z,
            'wmi' : self.wmi,
            
            't_bounds' : self.t_bounds,
            't_enc' : self.t_enc,
            't_query' : self.t_query,
            't_sat' : self.t_sat,
            't_total' : self.t_total,
            't_Z' : self.t_Z
            }

    def _check_satisfiability(self, formula):
        t0 = time()
        print("Checking SAT ...", end=" ")
        self.is_sat = is_sat(formula)
        self.t_sat = time() - t0        
        print(f"{self.is_sat} (t: {self.t_sat})\n")

    def _compute_omt_bounds(self, formula):
        """
        print("Computing tighter bounds via OMT.")
        if not is_sat(formula):
            print("UnSAT bounds")
            return None

        #opt = Optimizer("msat_incr")
        opt = Optimizer("msat_sua")
        opt.add_assertion(formula)
        bounds = []
        t0 = time()
        for v in self.sys_encoder.X:
            print(f"Minimizing {v}.")
            model, vmin = opt.optimize(MinimizationGoal(v))
            print(f"Maximizing {v}.")
            model, vmax = opt.optimize(MaximizationGoal(v))
            bounds.append((float(vmin.constant_value()),
                           float(vmax.constant_value())))

        t_omt = time() - t0
        return bounds
        """
        return self.loose_bounds



    def _compute_Z(self, support, weight, domain):
        print("Computing P(pre) ...", end=" ")
        self.solver = WMISolver(support, weight)
        t0 = time()
        Z, self.nint_Z = self.solver.computeWMI(Bool(True), domain,
                                                cache=self.cache)
        self.t_Z = time() - t0
        self.Z = Z
        print(f"Z: {Z} (nint: {self.nint_Z}, t: {self.t_Z})\n")
        return Z


    def _compute_query(self, bound, Z, query, partitions, domain):        
        assert(self.solver is not None), "should be set in _compute_partition(.)"
        self.bound_hit = False
        t_query = time()
        wmi_query = 0; self.nint_query = 0
        for n_part, part in enumerate(partitions):

            t0 = time()
            print(f" Computing [{n_part+1}/{len(partitions)}] ...", end=" ")
            wmi_part, nint_part = self.solver.computeWMI(And(query, part), domain, cache=self.cache)
            t_part = time() - t0
            
            wmi_query += wmi_part
            self.nint_query += nint_part

            print(f"[{wmi_query/Z} >= {bound}] (nint: {nint_part}, t: {t_part})", end=" ")

            if wmi_query >= bound * Z and n_part < len(partitions) - 1:
                print("!!BOUND HIT!!")
                self.bound_hit = True
                break
            else: print()

        print()
        self.t_query = time() - t_query
        self.wmi = wmi_query

        return wmi_query
        

    def _encode_system(self, bounds):

        if not self.use_ibp:
            bounds = None

        print("\nEncoding SYS", end=" ")
        t0 = time()
        sys_enc = self.sys_encoder.smt_formula(
            bounds=bounds
        )
        self.t_enc = time() - t0
        print(f"t: {self.t_enc}\n")


        return sys_enc

    def check_property(self, k, pre, post):

        t0 = time()
        support = And(self.prior.smt_formula(), pre)

        print("OMT bounds ...", end=" ")
        t1 = time()
        bounds = self._compute_omt_bounds(support)
        self.t_bounds = time() - t1
        print(f"Done in {self.t_bounds} s.")

        ##################################################

        Z = self._compute_Z(support,
                            self.prior.smt_weight(),
                            self.domain)

        ##################################################
        sys_enc = self._encode_system(bounds)
        ##################################################

        query = And(support, post, sys_enc)
        self._check_satisfiability(query)
        if not self.is_sat:
            wmi_query = 0.0
            self.nint_query = 0
            self.t_query = 0.0
            self.wmi = 0.0
            self.bound_hit = False

        else:
            partitions = self.sys_encoder.partition(support,
                                                    post,
                                                    bounds,
                                                    w=self.prior.evaluate)
            wmi_query = self._compute_query(k, Z,
                                            query,
                                            partitions,
                                            self.domain)

        self.t_total = time() - t0
        return wmi_query, Z


    def check_local_robustness(self, k, xc, yc, epsilon, delta, reverse=False):

        t0 = time()
        clauses = []
        for i, xi in enumerate(self.sys_encoder.X):
            clauses.append(LE(Real(float(xc[i] - epsilon)), xi))
            clauses.append(LE(xi, Real(float(xc[i] + epsilon))))

        pre = And(*clauses)
        clauses = []
        for i, yi in enumerate(self.sys_encoder.Y):
            if self.sys_encoder.model.discrete_output:
                if not reverse:
                    clauses.append(yi if bool(yc[i]) else Not(yi))
                else:
                    clauses.append(Not(yi) if bool(yc[i]) else yi)
            else:
                if not reverse:
                    clauses.append(And(LE(Real(float(yc[i] * (1 - delta))), yi),
                                       LE(yi, Real(float(yc[i] * (1 + delta))))))
                else:
                    clauses.append(Not(And(LE(Real(float(yc[i] * (1 - delta))), yi),
                                           LE(yi, Real(float(yc[i] * (1 + delta)))))))
   
        post = And(*clauses) 
        support = And(self.prior.smt_formula(), pre)

        t1 = time()
        print("Tightening bounds ...", end=" ")
        vol = lambda l : prod(lx[1] - lx[0] for lx in l)
        bounds = []
        for i in range(len(self.sys_encoder.X)):
            prev_lb, prev_ub = self.loose_bounds[i]
            loc_lb, loc_ub = xc[i] - epsilon, xc[i] + epsilon
            bounds.append([max(prev_lb, loc_lb),
                           min(prev_ub, loc_ub)])

        print(f"Volume: (before) {vol(self.loose_bounds)} (after) {vol(bounds)}")

        self.t_bounds = time() - t1

        ##################################################

        Z = self._compute_Z(support,
                            self.prior.smt_weight(),
                            self.domain)

        ##################################################

        sys_enc = self._encode_system(bounds)

        ##################################################
        query = And(support, post, sys_enc)
        self._check_satisfiability(query)
        if not self.is_sat:
            wmi_query = 0
            self.nint_query = 0
            self.t_query = 0.0
            self.wmi = 0.0
            self.bound_hit = False

        else:
            partitions = self.sys_encoder.partition(support,
                                                    post,
                                                    bounds,
                                                    w=self.prior.evaluate)
            wmi_query = self._compute_query(k, Z,
                                            query,
                                            partitions,
                                            self.domain)
        self.t_total = time() - t0
        return wmi_query, Z


    def check_local_equivalence(self, surrogate_encoder, k, xc, epsilon, reverse=False):

        t0 = time()
        clauses = []
        for i, xi in enumerate(self.sys_encoder.X):
            clauses.append(LE(Real(float(xc[i] - epsilon)), xi))
            clauses.append(LE(xi, Real(float(xc[i] + epsilon))))

        pre = And(*clauses)
        clauses = []
        for i in range(len(self.sys_encoder.Y)):
            yi1 = self.sys_encoder.Y[i]
            yi2 = surrogate_encoder.Y[i]
            if not reverse:
                clauses.append(Iff(yi1, yi2))
            else:
                clauses.append(Not(Iff(yi1, yi2)))
   
        post = And(*clauses) 
        support = And(self.prior.smt_formula(), pre)

        t1 = time()
        print("Tightening bounds ...", end=" ")
        vol = lambda l : prod(lx[1] - lx[0] for lx in l)
        bounds = []
        for i in range(len(self.sys_encoder.X)):
            prev_lb, prev_ub = self.loose_bounds[i]
            loc_lb, loc_ub = xc[i] - epsilon, xc[i] + epsilon
            bounds.append([max(prev_lb, loc_lb),
                           min(prev_ub, loc_ub)])

        print(f"Volume: (before) {vol(self.loose_bounds)} (after) {vol(bounds)}")

        self.t_bounds = time() - t1

        ##################################################

        Z = self._compute_Z(support,
                            self.prior.smt_weight(),
                            self.domain)

        ##################################################

        if not self.use_ibp:
            bounds = None

        print("\nEncoding SYS1 and SYS2", end=" ")
        t0 = time()
        sys1_enc = self.sys_encoder.smt_formula(
            bounds=bounds
        )
        sys2_enc = surrogate_encoder.smt_formula(
            bounds=bounds
        )
        self.t_enc = time() - t0
        print(f"t: {self.t_enc}\n")


        ##################################################

        query = And(support, post, sys1_enc, sys2_enc)
        self._check_satisfiability(query)
        if not self.is_sat:
            wmi_query = 0
            self.nint_query = 0
            self.t_query = 0.0
            self.wmi = 0.0
            self.bound_hit = False

        else:
            partitions = self.sys_encoder.partition(support,
                                                    post,
                                                    bounds,
                                                    w=self.prior.evaluate)
            wmi_query = self._compute_query(k, Z,
                                            query,
                                            partitions,
                                            self.domain)
        self.t_total = time() - t0
        return wmi_query, Z



    def check_monotonicity(self, k, input_index, output_index=0, bounds=None):
        
        t0 = time()

        sc_support = self_compose_formula(self.prior.smt_formula())
        sc_weight = self_compose_weight(self.prior.smt_weight())

        cloned_input = [clone_var(x) for x in self.sys_encoder.X]
        cloned_output = [clone_var(y) for y in self.sys_encoder.Y]

        hyperdomain = set(cloned_input).union(self.domain)

        pre = LE(self.sys_encoder.X[input_index], cloned_input[input_index])
        op = Implies if self.sys_encoder.model.discrete_output else LE
        post = op(self.sys_encoder.Y[output_index], cloned_output[output_index])
        support = And(self.prior.smt_formula(), pre)

        t1 = time()
        print("OMT bounds ...", end=" ")
        t1 = time()
        bounds = self._compute_omt_bounds(support)
        self.t_bounds = time() - t1
        print(f"Done in {self.t_bounds} s.")

        self.t_bounds = time() - t1

        ##################################################

        Z = self._compute_Z(support,
                            self.prior.smt_weight(),
                            hyperdomain)

        ##################################################

        sys_enc = self._encode_system(bounds)

        ##################################################

        query = And(support, post, sys_enc)
        self._check_satisfiability(query)
        if not self.is_sat:
            wmi_query = 0
            self.nint_query = 0
            self.t_query = 0.0
            self.wmi = 0.0
            self.bound_hit = False

        else:
            partitions = self.sys_encoder.partition(support,
                                                    post,
                                                    bounds,
                                                    w=self.prior.evaluate)
            wmi_query = self._compute_query(k, Z,
                                            query,
                                            partitions,
                                            hyperdomain)
        self.t_total = time() - t0
        return wmi_query, Z
        



