#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

for hidden_features in 128  # 32 64 128 256 512
do
    for learning_rate in 1e-3
    do
        for std in 0.2 0.4 0.8
        do
            for weight_decay in 1e-5 5e-5 1e-4 5e-4 1e-3
            do
                for depth in 2 3 4 # 5 6 7 8
                do
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:15 python run.py\
                --n_epochs 2000 \
                --n_samples 16\
                --depth $depth\
                --hidden_features $hidden_features \
                --weight_decay $weight_decay\
                --learning_rate $learning_rate \
                --std $std \
                --out $depth"_"$hidden_features"_"$learning_rate"_"$weight_decay"_"$std
            done
        done
    done
done
done
