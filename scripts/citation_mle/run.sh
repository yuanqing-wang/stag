#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 4:00

python run.py --n_epochs 200 --hidden_features 16
