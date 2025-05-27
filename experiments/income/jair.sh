
COMMON='--epochs 1000'

python3 generate_data.py
python3 run_experiment.py unbiased $COMMON
python3 run_experiment.py biased $COMMON
