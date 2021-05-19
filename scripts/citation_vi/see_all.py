import numpy as np

def get_traj(name):
    vl = np.load(
        "%s/accuracy_vl.npy" % name
    )

    te = np.load(
        "%s/accuracy_te.npy" % name
    )

    return te[vl.argmax()], vl[vl.argmax()]

def get_trajs(names):

    tes, vls = zip(*[get_traj(name) for name in names])
    return np.mean(tes), np.std(tes), np.mean(vls), np.std(vls)
    

if __name__ == "__main__":
    import os
    dirs = os.listdir('.')
    base_dirs = set(["_".join(_dir.split("_")[:-1]) for _dir in dirs])
    for base_dir in base_dirs:
        print(base_dir)
        try:
            names = []
            for name in dirs:
                if name.startswith(base_dir):
                    names.append(name)
            mean_te, std_te, mean_vl, std_vl = get_trajs(names)
            print("%.2f ± %.2f" % (mean_te*100, std_te*100), "%.2f ± %.2f" % (mean_vl*100, std_vl*100))

        except:
            pass

            
