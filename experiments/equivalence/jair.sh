

EPSILONS='0.01 0.05 0.1'
COMMON='--n_points 100 --k 0.1 --reverse_class --df_size 4 --data_folder JAIR'

for EPSILON in $EPSILONS
do
    python3 run_experiment.py $COMMON --epsilon $EPSILON --use_ibp --partitions sample-16
done

for EPSILON in $EPSILONS
do
    python3 run_experiment.py $COMMON --epsilon $EPSILON --use_ibp
done

python3 compute_plots.py
