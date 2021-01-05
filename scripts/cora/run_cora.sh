#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 128 256 512
do
    for act in "elu"  # 'sigmoid' # 'leaky_relu' 'tanh'
    do
        for layer in 'GraphConv' #'SAGEConv' 'GINConv' 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "none" "normal" "uniform"
                    do
                        for alpha in 0.01 0.1 0.2 0.4
                        do
                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 1:00 -R "rusage[mem=12] span[hosts=1]" -W 1:00 -o %J.stdout python run_cora.py --layer $layer --stag $stag --hidden_features $units --n_epochs 3000 --depth 6 --out $stag"_"$layer"_"$units"_"$act"_"$alpha"_depth6"


                done
        done
    done
done
done
