#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00
for repeat in {0..10}
do
        for std in 0.0 0.1 0.2 0.3 0.4 0.5 # .2 0.4 0.8 # 1.0 0.5 0.25
        do

            bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes:mode=shared" -R "rusage[mem=10] span[ptile=1]" -W 0:30 python run.py\
                --std $std \
                --out $std"_"$repeat
            done
        done

