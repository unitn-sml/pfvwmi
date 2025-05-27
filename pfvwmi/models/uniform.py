
import numpy as np
import pickle
import pysmt.shortcuts as smt


class UniformPrior:

    """This class implements a uniform prior."""

    def __init__(self, feats, real_bounds):
            self.feats = feats
            self.real_bounds = real_bounds

    def _reformat_data(self, data):
        assert(isinstance(data, np.ndarray)), "Expected numpy.ndarray"
        rdata = data[:, [i for i, f in enumerate(self.feats)
                         if f[1] == 'real']]

        bdata = data[:, [i for i, f in enumerate(self.feats)
                         if f[1] == 'bool']].astype(bool)

        return rdata, bdata

    ################################################## PROPERTIES
    @property
    def bfeats(self):
        return [n for n, t in self.feats if t == 'bool']

    @property
    def rfeats(self):
        return [n for n, t in self.feats if t == 'real']

    @property
    def volume(self):
        b = np.array(self.real_bounds)
        return 2 ** len(self.bfeats) * float(np.prod(b[:,1] - b[:, 0]))

    @property
    def weight(self):
        return 1 / self.volume

    def get_bounds(self, vnames=None):
        bounds = []
        for vi, vname in enumerate(self.rfeats):
            if vnames is None or vname in vnames:
                bounds.append(self.real_bounds[vi])

        return bounds

    def evaluate(self, x):
        """Computes the likelihood of a sample."""
        _, real_x = self._reformat_data(x)
        b = self.get_bounds()
        return self.weight \
            * np.all((real_x >= b[:,0]).astype(int) \
            * (real_x <= b[:,1]).astype(int), axis=1).astype(int)

    ################################################## SMT ENCODING

    def smt_variables(self, smt_env=None):
        """Returns the list of pysmt variables."""

        if smt_env is None: smt_env = smt.get_env()

        smt_vars = []
        for vname, vtype in self.feats:
            t = smt.BOOL if (vtype == 'bool') else smt.REAL
            var = smt_env.formula_manager.get_or_create_symbol(vname, t)
            smt_vars.append(var)

        return smt_vars

    def smt_formula(self, smt_env=None):
        """Returns a pysmt formula encoding the axis-aligned bounds."""

        if smt_env is None: smt_env = smt.get_env()

        smt_vars = self.smt_variables(smt_env)
        rvars = [v for v in smt_vars if v.symbol_type() == smt.REAL]
        clauses = []
        for i, smt_var in enumerate(rvars):
            lb, ub = self.real_bounds[i]
            clauses.append(smt.LE(smt.Real(float(lb)), smt_var))
            clauses.append(smt.LE(smt_var, smt.Real(float(ub))))

        return smt.And(*clauses)

    def smt_weight(self, smt_env=None, formula_var=None, bounds=None):
        return smt.Real(self.weight)


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
