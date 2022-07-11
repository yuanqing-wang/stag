for std in 1e-10 # 0.2 0.3 0.4 0.5
do
    for repeat in 0 # 1 2
    do
        for model in GAT
        do
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 0:30 python run.py\
                --std $std \
                --model $model \
                --out $model"_"$std"_"$repeat
    done
done
done
