#BSUB -q cpuqueue
#BSUB -o %J.stdout

for units in 128
do
    for act in "relu" # "elu" # "sigmoid" "relu" "tanh" "leaky_relu"
    do
        for layer in 'GraphConv' # 'SGConv' 'EdgeConv' 'GINConv'
        do
                    for stag in "vi"
                    do
                            for depth in 2
                            do
                                for repeat in 1 2 3 4 5
                                do
                                    for a_prior in 0.2 0.5 1.0
                                    do
                                        for a_log_sigma_init in 0 1 2
                                        do
                                            for a_mu_init_std in 0.25 0.5 1.0
                                            do
                                                for lr_vi in 1e-3
                                                do


                                                bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 0:03 -R "rusage[mem=4] span[hosts=1]" -o %J.stdout python run_cora.py --layer $layer --hidden_features $units --n_epochs 3000 --depth $depth --lr 0.005 --a_prior $a_prior --a_log_sigma_init $a_log_sigma_init --a_mu_init_std $a_mu_init_std --lr_vi $lr_vi --act $act --out "vi_depth_"$depth"_"$a_prior"_"$a_log_sigma_init"_"$a_mu_init_std"_"$lr_vi"_"$repeat

                done
        done
    done
done
done
done
done
done
done
done


