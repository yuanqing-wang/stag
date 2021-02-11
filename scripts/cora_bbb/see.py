import numpy as np

def get_traj(stag="none", alpha=1.0, depth=4, repeat=0):
    vl = np.load(
        "%s_GraphConv_128_relu_%s_depth_%s_repeat%s/accuracy_vl.npy" % (stag, alpha, depth, repeat)
    )

    te = np.load(
        "%s_GraphConv_128_relu_%s_depth_%s_repeat%s/accuracy_te.npy" % (stag, alpha, depth, repeat)
    )

    return te[vl.argmax()]

def get_trajs(stag="none", alpha=1.0, depth=4):

    xs = np.array([get_traj(stag=stag, alpha=alpha, depth=depth, repeat=repeat) for repeat in [1, 2, 3, 4, 5]])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    stag = sys.argv[1]
    alpha = sys.argv[2]
    depth = sys.argv[3]
    mean, std = get_trajs(stag, alpha, depth)
    print("%.2f ± %.2f" % (mean*100, std*100))


