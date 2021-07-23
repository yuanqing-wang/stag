#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

for hidden_features in 32 64 128
do
    for learning_rate in 1e-3 1e-4 1e-5
    do
        for std in 1.0 0.5 0.25
        do
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:15 python run.py\
                --n_epochs 2000 \
                --hidden_features $hidden_features \
                --learning_rate $learning_rate \
                --std $std \
                --out $hidden_features"_"$learning_rate"_"$std
            done
        done
    done


