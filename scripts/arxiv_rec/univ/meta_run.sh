#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

for std in 0.0 0.1 0.2 0.3 0.4 0.5  # 0.2 0.4 0.8
do
                for data in cora# citeseer cora pubmed
                do
                    for repeat in 0 1 2 
                    do
                        for model in GCN
                        do
                            for distribution in Normal # Uniform
                            do

                        out=$model"_"$distribution"_"$data"_"$std"_"$repeat
                        bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:15 python run.py\
        --model $model \
        --std $std \
        --data $data \
        --distribution $distribution \
        --out $out
            done
        done
    done
done
done
