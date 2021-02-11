import numpy as np

def get_traj(stag="none", alpha=1.0, depth=4, repeat=0, model="GraphConv"):
    vl = np.load(
        "%s_%s_128_relu_%s_depth_%s_repeat%s/accuracy_vl.npy" % (stag, model, alpha, depth, repeat)
    )

    te = np.load(
        "%s_%s_128_relu_%s_depth_%s_repeat%s/accuracy_te.npy" % (stag, model, alpha, depth, repeat)
    )

    return te[vl.argmax()]

def get_trajs(stag="none", alpha=1.0, depth=4, model="GraphConv"):

    xs = np.array([get_traj(stag=stag, alpha=alpha, depth=depth, repeat=repeat, model=model) for repeat in range(1, 6)])

    return np.mean(xs), np.std(xs)

if __name__ == "__main__":
    import sys
    stag = sys.argv[1]
    alpha = sys.argv[2]
    depth = sys.argv[3]

    if len(sys.argv) >= 5:
        model = sys.argv[4]
    else:
        model = "GraphConv"

    mean, std = get_trajs(stag, alpha, depth, model)
    print("%.2f ± %.2f" % (mean*100, std*100))


