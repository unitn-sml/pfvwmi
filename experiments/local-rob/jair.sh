

EPSILONS1='0.01 0.05 0.1'
EPSILONS2='0.1 0.15 0.2'

COMMON='--n_points 100 --k 0.1 --reverse_robustness --data_folder JAIR'
SYS='nn --nn_hidden 32 32 32'

for EPSILON in $EPSILONS1
do
    python3 run_experiment.py $SYS $COMMON --epsilon $EPSILON 
    python3 run_experiment.py $SYS $COMMON --epsilon $EPSILON --use_ibp
done

for EPSILON in $EPSILONS2
do
    python3 run_experiment.py $SYS $COMMON --epsilon $EPSILON --use_ibp
    python3 run_experiment.py $SYS $COMMON --epsilon $EPSILON --use_ibp --partitions sample-16
done

python3 compute_tables.py
python3 compute_plots.py
