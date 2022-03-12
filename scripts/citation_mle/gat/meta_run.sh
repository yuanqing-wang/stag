#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

        for std in 0.0 # 0.1 0.2 0.3 0.4 0.5  # 0.2 0.4 0.8
        do
                        for data in cora citeseer pubmed
                        do
                            for repeat in 0 1 2 3 4 
                            do
                                for model in GAT
                                do
            out=$model"_"$data"_"$std"_"$repeat
            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 0:15 python run.py\
                --model $model \
                --std $std \
                --data $data \
                --out $out
            done
        done
    done
done
