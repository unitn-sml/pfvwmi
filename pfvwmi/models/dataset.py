
import numpy as np
import pickle

import pysmt.shortcuts as smt


class Dataset:

    """Generic class implementing a dataset."""

    def __init__(self, varnames, vartypes, values):
        self.varnames = list(varnames)
        self.vartypes = list(vartypes)
        self.values = np.array(values)

        assert(len(varnames) == len(vartypes))
        assert(len(varnames) == self.values.shape[1])

    ################################################## UTILS


    def split(self, ratio):
        """Returns two instances containing respectively
        ratio and (1-ratio) of the original instances."""

        assert(0 < ratio and ratio < 1)
        
        size_first = int(self.size[0] * ratio)
        d1 = Dataset(self.varnames,
                     self.vartypes,
                     self.values[:size_first])
        d2 = Dataset(self.varnames,
                     self.vartypes,
                     self.values[size_first:])

        return d1, d2
        
        
    ################################################## FILE I/O
    
    def dump(self, path):
        """Dump the dataset to file."""

        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load(path):
        """Load the dataset from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


    ################################################## SMT ENCODING


    def smt_variables(self, smt_env=None):
        """Returns the list of pysmt variables."""

        if smt_env is None: smt_env = smt.get_env()

        smt_vars = []
        for i, name in enumerate(self.varnames):
            if self.vartypes[i] == 'real':
                smt_type = smt.REAL
            elif self.vartypes[i] == 'bool':
                smt_type = smt.BOOL
            else:
                raise NotImplementedError()

            var = smt_env.formula_manager.get_or_create_symbol(name, smt_type)
            smt_vars.append(var)

        return smt_vars


    def smt_bounds(self, smt_env=None):
        """Returns a pysmt formula encoding the axis-aligned bounds."""

        l_bounds = np.min(self.values, axis=0)
        u_bounds = np.max(self.values, axis=0)

        clauses = []
        for i, smt_var in enumerate(self.smt_variables(smt_env)):
            clauses.append(smt.LE(Real(l_bounds[i]), smt_var))
            clauses.append(smt.LE(smt_var, Real(u_bounds[i])))

        return smt.And(*clauses)

    @property
    def size(self):
        return self.values.shape
            

        
        
