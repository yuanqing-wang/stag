#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

        for std in 0.1 # 0.0 0.1 0.2 0.3 0.4 0.5  # 0.2 0.4 0.8
        do
                        for data in cora # citeseer cora pubmed
                        do
                            for repeat in 0
                            do
                                for kl_scaling in 1.0 # 0.001 0.0001 0.01 0.1 # 1.0
                                do
                                    for model in GCN
                                    do
            out=$data"_"$model"_"$std"_"$repeat
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=2] span[ptile=1]" -W 0:30 python run.py\
                --model $model \
                --data $data \
                --std $std \
                --out $out
            done
        done
    done
done
done

