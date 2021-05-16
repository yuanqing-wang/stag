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
                            for depth in 2 # 3 4 
                            do
                                for repeat in 1 # 2 3 4 5 
                                do
                                    for a_prior in 0.75
                                    do
                                        for a_log_sigma_init in 3 4 2 1 0 -1 -2 -3 -4 -5  
                                        do
                                            for a_mu_init_std in 0.1 0.25 0.5 1.0 1.5
                                            do

                                                for data in "cora"
                                                do
                                                    bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -n 1 -W 0:10 -R "rusage[mem=8] span[hosts=1]" -o %J.stdout python run.py --layer $layer --data $data --hidden_features $units --n_epochs 1000 --depth $depth --lr 1e-3 --a_prior $a_prior --act $act --a_log_sigma_init $a_log_sigma_init --a_mu_init_std $a_mu_init_std --out "vi_depth_"$depth"_"$data"_"$a_prior"_"$a_log_sigma_init"_"$a_mu_init_std"_"$repeat

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
