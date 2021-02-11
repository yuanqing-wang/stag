import numpy as np

def get_traj(name, repeat):
    vl = np.load(
        "name_repeat%s/accuracy_vl.npy" % (name, repeat)
    )

    te = np.load(
        "name_repeat%s/accuracy_te.npy" % (name, repeat)
    )

    return te[vl.argmax()]

def get_trajs(name):

    xs = np.array([get_traj(name, repeat=repeat) for repeat in [1, 2, 3, 4, 5]])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    stag = sys.argv[1]
    alpha = sys.argv[2]
    depth = sys.argv[3]
    mean, std = get_trajs(stag, alpha, depth)
    print("%.2f ± %.2f" % (mean*100, std*100))


