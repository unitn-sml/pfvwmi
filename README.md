# pfvwmi
Probabilistic formal verification via Weighted Model Integration

## Installation

1. Clone and install the repository:
   ```
   cd pfvwmi/
   pip3 install -e .
   ```

2. Install the `MathSAT` SMT solver via `pysmt`:
   ```
   pysmt-install --msat --confirm-agreement
   ```

## Running the experiments

Note: some numerical results might slightly differ from the ones presented in
the paper due to [non-determinism](https://docs.pytorch.org/docs/stable/notes/randomness.html)
in `pytorch`. The conclusions and take-aways should hold true.

### Income

   ```
   cd experiments/income/
   bash jair.sh
   ```

### Cartpole

   ```
   cd experiments/cartpole/
   python3 run_experiment.py
   ```

### Probabilistic local robustness

   ```
   cd experiments/local-rob/
   bash jair.sh
   ```

### Probabilistic equivalence (NN vs. RF)

   ```
   cd experiments/equivalence/
   bash jair.sh
   ```


