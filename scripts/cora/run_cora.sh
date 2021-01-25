#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 32 64 128 256
do
    for act in "relu" # "elu" # "sigmoid" "relu" "tanh" "leaky_relu"
    do
        for layer in 'GINConv' # 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "none" "normal" "uniform" "bernoulli"
                    do
                        for alpha in 0.05 0.1 0.2 0.4
                        do
                            for depth in 3 4 5 6 7 8
                            do
                                for repeat in {1..5}
                                do
                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 1:00 -R "rusage[mem=12] span[hosts=1]" -W 1:00 -o %J.stdout python run_cora.py --layer $layer --stag $stag --hidden_features $units --n_epochs 3000 --depth $depth --alpha $alpha --lr 0.005 --out $stag"_"$layer"_"$units"_"$act"_"$alpha"_depth_"$depth"_repeat"$repeat

                done
        done
    done
done
done
done
done

