import numpy as np

def get_traj(name, repeat):
    vl = np.load(
        "%s_%s/accuracy_vl.npy" % (name, repeat)
    )

    te = np.load(
        "%s_%s/accuracy_te.npy" % (name, repeat)
    )

    return te[vl.argmax()]

def get_trajs(name):

    xs = np.array([get_traj(name, repeat=repeat) for repeat in [1, 2, 3, 5]])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    mean, std = get_trajs(name)
    print("%.2f ± %.2f" % (mean*100, std*100))


