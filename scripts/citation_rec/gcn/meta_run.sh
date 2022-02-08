#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

for hidden_features in 16 # 32 64 128 256 512
do
    for learning_rate in 1e-2 # 1e-3 1e-4 1e-5
    do
        for std in 0.1 # 0.2 0.3 0.4 0.5  # 0.2 0.4 0.8
        do
                for depth in 2 # 5 6 7 8
                do
                    for weight_decay in 5e-4
                    do
                        for data in cora # citeseer cora pubmed
                        do
                            for repeat in 0 # 1 2 
                            do
                                for kl_scaling in 1.0
                                do
            out=$data"_"$depth"_"$hidden_features"_"$learning_rate"_"$weight_decay"_"$std"_"$repeat
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:15 python run.py\
                --weight_decay $weight_decay \
                --n_epochs 2000 \
                --n_samples 4 \
                --n_samples_training 1 \
                --depth $depth\
                --hidden_features $hidden_features \
                --learning_rate $learning_rate \
                --std $std \
                --data $data \
                --out $out \
                --kl_scaling $kl_scaling
            done
        done
    done
done
done
done
done
done
