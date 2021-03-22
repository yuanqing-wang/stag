#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 128
do
    for act in "relu" # "elu" # "sigmoid" "relu" "tanh" "leaky_relu"
    do
        for layer in 'GraphConv' # 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "none" "normal"  # "none_de" "none_dropout" "none_gdc"
                    do
                        for alpha in 0.2 0.4 0.8
                        do
                            for depth in 2
                            do
                                for repeat in {1..5}
                                do
                                    for data in ogbn-arxiv
                                    do
                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 0:15 -R "rusage[mem=12] span[hosts=1]" -W 1:00 -o %J.stdout python run_cora.py --data $data --layer $layer --stag $stag --hidden_features $units --n_epochs 3000 --depth $depth --alpha $alpha --lr 0.005 --out $stag"_"$layer"_"$units"_"$act"_"$alpha"_depth_"$depth"_data"$data"_repeat"$repeat

                done
        done
    done
done
done
done
done
done

