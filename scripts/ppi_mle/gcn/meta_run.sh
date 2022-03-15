for std in 0.0 0.01 0.02 0.03 0.04 0.05 # 0.0 0.1 0.2 0.3 0.4 0.5
do
    for repeat in {0..1}
    do
        for model in GCN GraphSAGE
        do
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 0:59 python run.py\
                --std $std \
                --model $model \
                --out $model"_"$std"_"$repeat
    done
done
done
