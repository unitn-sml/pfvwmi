
from copy import deepcopy
import numpy as np
import pickle
import pysmt.shortcuts as smt


class Node:

    def __init__(self, tree, parent, data, bounds):

        # structure
        self.tree = tree
        self.parent = parent
        self.children = None
        self.split_var = None
        self.split_val = None

        # node quantities
        self.bounds = bounds        
        self.vol = DET._compute_vol(*bounds)

        self.n_inst = len(data[0])
        self._grow(data)

    ################################################## PROPERTIES

    @property
    def weight(self):
        return self.n_inst / (self.tree.N * self.vol)

    @property
    def R(self):
        return self.tree._compute_R(self.n_inst, self.vol)

    @property
    def is_leaf(self):
        """Is this a leaf?."""
        return (self.children is None)

    @property
    def is_root(self):
        """Is this the root?."""
        return (self.parent is None)

    @property
    def leaves(self):
        """Returns the set of leaves in the (sub)tree."""
        if self.is_leaf:
            return {self}
        else:
            return set.union(self.children[0].leaves,
                             self.children[1].leaves)
        
    @property
    def prune_points(self):
        """Returns the list of prune points."""
        if self.is_leaf:
            return []
        elif (self.children[0].is_leaf and
              self.children[1].is_leaf and
              self.n_inst < self.tree.n_max):
            return [self]
        else:
            return self.children[0].prune_points + \
                self.children[1].prune_points
            

    ################################################## INTERNALS

    def _grow(self, data):

        def partition(data, split):
            """Binary data split based on feature/value."""
            rd, bd = deepcopy(data)
            feat, val = split
            if val is None: # boolean split
                rd_l, rd_r = rd[bd[:,feat]], rd[~bd[:,feat]]
                bd_l, bd_r = bd[bd[:,feat]], bd[~bd[:,feat]]
            else: # numerical split
                rd_l, rd_r = rd[rd[:,feat] <= val], rd[rd[:,feat] > val]
                bd_l, bd_r = bd[rd[:,feat] <= val], bd[rd[:,feat] > val]

            return (rd_l, bd_l), (rd_r, bd_r)

        def new_bounds(split):
            """Computes new left/right bounds based on split."""
            bounds_l = deepcopy(self.bounds)
            bounds_r = deepcopy(self.bounds)

            i, val = split

            if val is None:
                bounds_l[1][0][i] = True
                #bounds_l[1][1][i] = True
                #bounds_r[1][0][i] = False
                bounds_r[1][1][i] = False
            else:
                bounds_l[0][1][i] = val
                bounds_r[0][0][i] = val

            return bounds_l, bounds_r
            

        def score_split(n_l, n_r, bounds_l, bounds_r):

            vol_l = DET._compute_vol(*bounds_l)
            vol_r = DET._compute_vol(*bounds_r)

            assert(n_l + n_r == self.n_inst)
            assert(np.isclose(vol_l + vol_r, self.vol))
            
            R_l = self.tree._compute_R(n_l, vol_l)
            R_r = self.tree._compute_R(n_r, vol_r)
            score = abs(self.R - (R_l + R_r))

            return score


        if self.n_inst > self.tree.n_min:

            # we need to find the best s* = (feat/value) for splitting
            # i.e. the one that reduces the reduction in R(t):
            # argmax_s [R(t) - R(tl) - R(tr)]
            best_score, best_split = None, None
            best_partition, best_bounds = None, None

            # boolean splits
            splits = []
            #[(i, None) for i, _ in enumerate(self.tree.bfeats)]
            for i, _ in enumerate(self.tree.bfeats):
                if len(np.unique(data[1][:, i])) > 1:
                    splits.append((i, None))

            # numerical splits
            for i, _ in enumerate(self.tree.rfeats):
                suvals = sorted(np.unique(data[0][:, i]))
                candidate_vals = [(suvals[i]+suvals[i+1])/2
                                  for i in range(len(suvals)-1)]
                for val in candidate_vals:
                    splits.append((i, val))
            
            for split in splits:
                data_l, data_r = partition(data, split)
                n_l, n_r = len(data_l[0]), len(data_r[0])
                bounds_l, bounds_r = new_bounds(split)

                if (n_l == 0) or (n_r == 0):
                    continue

                score = score_split(n_l, n_r, bounds_l, bounds_r)
                
                if (best_score is None) or (score > best_score):
                    best_score, best_split = score, split
                    best_partition = (data_l, data_r)
                    best_bounds = (bounds_l, bounds_r)
            
            left_child = Node(self.tree, self,
                              best_partition[0],
                              best_bounds[0])

            right_child = Node(self.tree, self,
                               best_partition[1],
                               best_bounds[1])

            self.children = [left_child, right_child]
            self.split_var, self.split_val = best_split

    def _eval(self, sample):
        """Computes the likelihood of a sample."""

        if self.is_leaf:
            wsample = np.ones(len(sample[0])) * self.weight
            return wsample
        else:            
            if self.split_val == None:
                cond = sample[1][:, self.split_var]
            else:
                cond = (sample[0][:, self.split_var] <= self.split_val)

            return (self.children[0]._eval(sample) * cond.astype(int) +
                    self.children[1]._eval(sample) * (~cond).astype(int))


    def _prune_node(self):
        assert(not self.is_leaf)
        self.children = None
        self.split_var = None
        self.split_val = None


class DET:

    """This class implements Density Estimation Trees (Ram et al. 2011)."""

    def __init__(self, feats, data=None, n_min=None, n_max=None):

        if data is not None:
            if n_min is None:
                n_min = max(1, len(data)//100) # 0.01 |D|
            if n_max is None:
                n_max = max(2 * n_min, len(data)//10) # 0.1 |D|

            assert(n_min < n_max)
        
            self.n_min = n_min # min. internal node size (regulates growing)
            self.n_max = n_max # max. leaf node size (regulates pruning)
            self.feats = feats
            self.N = len(data)

            rdata, bdata = self._reformat_data(data)

            real_bounds = np.min(rdata, axis=0), np.max(rdata, axis=0)
            bool_bounds = np.min(bdata, axis=0), np.max(bdata, axis=0)

            # grow tree
            self.root = Node(self, None, (rdata, bdata), (real_bounds, bool_bounds))


    ################################################## PROPERTIES


    @property
    def bfeats(self):
        return [n for n, t in self.feats if t == 'bool']

    @property
    def rfeats(self):
        return [n for n, t in self.feats if t == 'real']

    @property
    def size(self):
        def count(node):
            if node.is_leaf: return 1
            else: return 1 + count(node.children[0]) + count(node.children[1])

        return count(self.root)

    def get_bounds(self, vnames=None):
        bounds = []
        for vi, vname in enumerate(self.rfeats):
            if vnames is None or vname in vnames:
                lb = self.root.bounds[0][0][vi]
                ub = self.root.bounds[0][1][vi]
                bounds.append([lb, ub])

        return bounds


    ################################################## TRAINING/INFERENCE


    def prune(self, valid, max_iter=100):

        if self.root.is_leaf:
            return

        def postorder(node):
            if node.is_leaf: return []
            return postorder(node.children[0]) + \
                postorder(node.children[1]) + [node]

        def signature(tree):
            return ','.join(map(lambda n : str(n.label),
                                postorder(tree)))

        # uniquely label internal nodes (postorder fashion)
        for i, n in enumerate(postorder(self.root)):
            n.label = i

        refdata = self._reformat_data(valid)

        unique_structures = {signature(self.root)}
        to_prune = [deepcopy(self.root)]
        scored = []
        i = 0
        while not len(to_prune) == 0 and i < max_iter:
            i += 1            
            cur = to_prune.pop(0)
            score = np.sum(np.log(cur._eval(refdata)))
            n_leaves = len(cur.leaves)
            scored.append((cur, score))
            for p in range(len(cur.prune_points)):
                nxt = deepcopy(cur)
                nxt.prune_points[p]._prune_node()
                s_nxt = signature(nxt)
                if s_nxt not in unique_structures:
                    unique_structures.add(s_nxt)
                    to_prune.append(nxt)
 
        best = sorted(scored, key=lambda x : x[-1])[-1]
        self.root = best[0]



    def evaluate(self, x):
        """Computes the likelihood of a sample."""
        return self.root._eval(self._reformat_data(x))
    """\
            * np.all((x >= b[:,0]).astype(int) \
            * (x <= b[:,1]).astype(int), axis=1).astype(int)
    """


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
        bvars = [v for v in smt_vars if v.symbol_type() == smt.BOOL]

        clauses = []
        for i, smt_var in enumerate(rvars):
            clauses.append(smt.LE(smt.Real(float(self.root.bounds[0][0][i])),
                                  smt_var))
            clauses.append(smt.LE(smt_var,
                                  smt.Real(float(self.root.bounds[0][1][i]))))
        for i, smt_var in enumerate(bvars):
            if self.root.bounds[1][0][i]:
                clauses.append(smt_var)
            elif not self.root.bounds[1][1][i]:
                clauses.append(smt.Not(smt_var))
            else:
                clauses.append(smt.Or(smt_var, smt.Not(smt_var)))

        return smt.And(*clauses)

    def smt_weight(self, smt_env=None, formula_var=None, bounds=None):

        if smt_env is None: smt_env = smt.get_env()

        def enc_node(node, rvars, bvars, bounds, formula_var):
            if node.is_leaf:
                const_term = smt.Real(float(node.weight))
                if formula_var is not None:
                    return smt.Equals(formula_var, const_term)
                else:
                    return const_term
            else:
                
                if (bounds is not None) and (node.split_val is not None):
                    if bounds[node.split_var][0] > node.split_val:
                        return enc_node(node.children[1],
                                        rvars, bvars, bounds, formula_var)
                    elif bounds[node.split_var][1] < node.split_val:
                        return enc_node(node.children[0],
                                        rvars, bvars, bounds, formula_var)

                if node.split_val is None:
                    cond = bvars[node.split_var]
                else:
                    cond = smt.LE(rvars[node.split_var],
                                  smt.Real(float(node.split_val)))
                        
                return smt.Ite(cond,
                               enc_node(node.children[0],
                                        rvars, bvars, bounds, formula_var),
                               enc_node(node.children[1],
                                        rvars, bvars, bounds, formula_var))

        smt_vars = self.smt_variables(smt_env)
        rvars = [v for v in smt_vars if v.symbol_type() == smt.REAL]
        bvars = [v for v in smt_vars if v.symbol_type() == smt.BOOL]
        structure = enc_node(self.root, rvars, bvars, bounds, formula_var)

        if formula_var is None:
            return structure
        else:
            return smt.And(structure,
                           smt.LE(smt.Real(0), formula_var),
                           smt.Implies(smt.Not(self.smt_formula()),
                                       smt.Equals(formula_var, smt.Real(0))))


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    ################################################## INTERNALS

    def _compute_R(self, dt, vt):
        return -(dt ** 2) / (self.N ** 2 * vt)

    @staticmethod
    def _compute_vol(real_bounds, bool_bounds):
        rvol = np.prod(real_bounds[1] - real_bounds[0])
        bvol = 2 ** np.sum(bool_bounds[1].astype(int) \
                           - bool_bounds[0].astype(int))
        return rvol * bvol

    def _reformat_data(self, data):
        assert(isinstance(data, np.ndarray)), "Expected numpy.ndarray"
        rdata = data[:, [i for i, f in enumerate(self.feats)
                         if f[1] == 'real']]

        bdata = data[:, [i for i, f in enumerate(self.feats)
                         if f[1] == 'bool']].astype(bool)

        return rdata, bdata

    def to_leaf(self):
        leaf_det = DET(None)
        leaf_det.feats = deepcopy(self.feats)
        leaf_det.n_min = self.n_min
        leaf_det.n_max = self.n_max
        leaf_det.N = self.N
        leaf_root = deepcopy(self.root)
        leaf_root.children = None
        leaf_root.split_var = None
        leaf_root.split_val = None
        leaf_det.root = leaf_root
        return leaf_det


if __name__ == '__main__':

    from test import generate_problem

    nreal = 2
    nbool = 2
    train = generate_problem(nreal, nbool, nbool > 0)
    valid_test = generate_problem(nreal, nbool, nbool > 0)
    valid = valid_test[:len(valid_test)//2]
    test = valid_test[len(valid_test)//2:]

    feats = [(f'x_{i}', 'real') for i in range(nreal)] + \
        [(f'b_{i}', 'bool') for i in range(nbool)]

    n_min = len(train) // 10
    n_max = len(train) // 2

    print(f"Training DET({n_min},{n_max})")
    det = DET(feats, train, n_min=n_min, n_max=n_max)

    print("L(test) before pruning:", np.mean(np.log(det.evaluate(test))))
    print("Size before pruning:", det.size)

    det.prune(valid)
    print("L(test) after pruning:", np.mean(np.log(det.evaluate(test))))
    print("Size after pruning:", det.size)

