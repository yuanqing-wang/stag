#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 32 64 128 256
do
    for act in "ReLU"  # 'sigmoid' # 'leaky_relu' 'tanh'
    do
        for layer in "GraphConv" # 'SAGEConv' 'GINConv' # 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "none" "normal" "uniform" "bernoulli"
                    do
                        for alpha in 0.05 0.1 0.2 0.4 # 0.8
                        do
                            for repeat in {0..5}
                            do
                                for depth in 3 4 5 6 7 8
                                do
                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -R "rusage[mem=12] span[hosts=1]" -W 1:00 -o %J.stdout python run.py --layer $layer --stag $stag --hidden_features $units --n_epochs 5000 --depth 8 --out $stag"_"$layer"_"$units"_"$act"_"$alpha"_depth"$depth"_repeat"$repeat

                done
        done
    done
done
done
done
done
