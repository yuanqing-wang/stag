#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 16
do
    for act in "relu" # "elu" # "sigmoid" "relu" "tanh" "leaky_relu"
    do
        for layer in 'GraphConv' # 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "normal" # "uniform" "bernoulli"
                    do
                        for alpha in 1.0
                        do
                            for depth in {1..10}
                            do
                                for repeat in {1..5}
                                do
                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 0:10 -R "rusage[mem=12] span[hosts=1]" -W 1:00 -o %J.stdout python run_cora.py --layer $layer --stag $stag --hidden_features $units --n_epochs 400 --depth $depth --alpha $alpha --lr 0.01 --out $stag"_"$layer"_"$units"_"$act"_"$alpha"_depth_"$depth"_repeat"$repeat

                done
        done
    done
done
done
done
done

