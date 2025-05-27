
import numpy as np
from pysmt.shortcuts import And, Equals, Iff, Implies, Ite, LE, Not, Times, \
    BOOL, REAL, get_env, serialize




def _op(f):
    if f.is_and(): return 'and'
    elif f.is_or(): return 'or'
    elif f.is_not(): return 'not'
    elif f.is_ite() : return 'ite'
    else:
        raise NotImplementedError("can't prettify", f)
            
def pretty_print(f, depth=0, k=2):
    if f.is_theory_relation() or len(f.args()) == 0: return serialize(f)
    else: return ('\n' + ' ' * depth + _op(f) + ' ') +\
        ('\n' + ' ' * depth + _op(f) + ' ').join([pretty_print(c, depth+k, k)
                                                 for c in f.args()])



################################################## SHORTCUTS


def Itex(a, b, c):
    """If-then-else with mutually exclusive bodies."""
    return And(Ite(a, b, c), Not(And(b, c)))


################################################## SELF-COMPOSITION


def clone_var(var, smt_env=None):
    """Clones a pysmt variable."""

    clone_name = lambda old : f'{old}_cl'
    
    if smt_env is None: smt_env = get_env()

    vname = var.symbol_name()
    vtype = var.symbol_type()

    return smt_env.formula_manager.get_or_create_symbol(
        clone_name(vname), vtype)


def clone_expression(expr, smt_env=None):
    """Clones a pysmt expression."""
    
    if smt_env is None: smt_env = get_env()

    return expr.substitute({v : clone_var(v, smt_env)
                            for v in expr.get_free_variables()})


def self_compose_formula(f, smt_env=None):
    """Applies self-composition on formula."""

    if smt_env is None: smt_env = get_env()

    return And(f, clone_expression(f, smt_env))


def self_compose_weight(w, smt_env=None):
    """Applies self-composition on weight."""

    if smt_env is None: smt_env = get_env()

    return Times(w, clone_expression(w, smt_env))


################################################## MONOTONICITY


def monotonicity(mx, xs, y, smt_env=None):
    """Returns the pre/post-conditions encoding monotonicity:

    (mx <= mx') => f(mx, xs) <= f(mx', xs)

    """

    assert(mx not in xs)

    if smt_env is None: smt_env = get_env()

    # precondition mx < mx', xs have the same value
    clauses = [LE(mx, clone_var(mx, smt_env))] 
    for var in xs:
        vtype = var.symbol_type()
        if vtype == REAL:
            clauses.append(Equals(var, clone_var(var, smt_env)))
        elif vtype == BOOL:
            clauses.append(Iff(var, clone_var(var, smt_env)))
        else:
            raise NotImplementedError()

    pre = And(*clauses)

    ytype = y.symbol_type()
    if ytype == REAL:
        post = LE(y, clone_var(y, smt_env))
    elif ytype == BOOL:
        post = Implies(y, clone_var(y, smt_env))
    else:
        raise NotImplementedError()

    return pre, post


        
    
