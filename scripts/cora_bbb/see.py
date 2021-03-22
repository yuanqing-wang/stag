import numpy as np

def get_traj(name, repeat):
    vl = np.load(
        "%s%s/accuracy_vl.npy" % (name, repeat)
    )

    te = np.load(
        "%s%s/accuracy_te.npy" % (name, repeat)
    )

    return te[vl.argmin()]

def get_trajs(name):

    xs = np.array([get_traj(name, repeat=repeat) for repeat in [1, 2, 3, 4, 5]])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    mean, std = get_trajs(name)
    print("%.4f ± %.4f" % (mean, std))


