import numpy as np

def get_traj(stag="none", alpha=1.0, depth=4, repeat=0, data="ESOL"):
    vl = np.load(
        "%s_%s_GraphConv_128_ReLU_%s_depth%s_repeat%s/accuracy_vl.npy" % (data, stag, alpha, depth, repeat)
    )

    te = np.load(
        "%s_%s_GraphConv_128_ReLU_%s_depth%s_repeat%s/accuracy_te.npy" % (data, stag, alpha, depth, repeat)
    )

    return te[vl.argmin()]

def get_trajs(stag="none", alpha=1.0, depth=4, data="ESOL"):

    xs = np.array([get_traj(stag=stag, alpha=alpha, depth=depth, repeat=repeat) for repeat in range(1, 6)])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    stag = sys.argv[1]
    alpha = sys.argv[2]
    depth = sys.argv[3]
    data =sys.argv[4]
    mean, std = get_trajs(stag, alpha, depth, data)
    print("%.2f ± %.2f" % (mean*100, std*100))


