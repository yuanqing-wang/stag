import json
import statistics

def run(name):
    import os
    names = [_name for _name in os.listdir() if name in _name and ".json" in _name]
    performances = []
    for name in names:
        with open(name, "r") as file_handle:
            performances.append(json.load(file_handle))
    accuracy_vl = [performance['accuracy_vl'] for performance in performances]
    accuracy_te = [performance['accuracy_te'] for performance in performances]
    
    accuracy_vl_mean = statistics.mean(accuracy_vl)
    accuracy_vl_std = statistics.stdev(accuracy_vl)
    
    accuracy_te_mean = statistics.mean(accuracy_te)
    accuracy_te_std = statistics.stdev(accuracy_te)

    print("vl: %.4f ± %.4f" % (accuracy_vl_mean, accuracy_vl_std))
    print("te: %.4f ± %.4f" % (accuracy_te_mean, accuracy_te_std))




if __name__ == "__main__":
    import sys
    run(sys.argv[1])
