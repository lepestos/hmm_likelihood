OBS_FILE = "obs"
OBS_LIKELIHOOD_FILE = "obs_ld"
TRANSITION_FILE = "trans_prob"
INITIAL_DIST_FILE = "init_prob"


def parse_data():
    """
    obs_ld is a matrix,
    where obs_ld[i][j] is the probabilty of the observed event i given hidden event j.
    We implement the following encoding for the observed events:
    1 -> 0
    2 -> 1
    3 -> 2
    and hidden events:
    COLD -> 0
    HOT -> 1
    """
    with open(OBS_FILE, "r") as f:
        obs = [int(c) - 1 for c in f.read().strip()]
    with open(OBS_LIKELIHOOD_FILE, "r") as f:
        obs_ld = [[float(c) for c in row.split()] for row in f.read().strip().split("\n")]
    with open(TRANSITION_FILE, "r") as f:
        trans_prob = [[float(c) for c in row.split()] for row in f.read().strip().split("\n")]
    with open(INITIAL_DIST_FILE, "r") as f:
        init_prob = [float(c) for c in f.read().split()]
    return obs, obs_ld, init_prob, trans_prob


def compute_forward(obs, obs_ld, init_prob, trans_prob):
    N, T = len(trans_prob), len(obs)
    forward = [[None] * T for _ in range(N)]
    
    for s in range(N):
        forward[s][0] = init_prob[s] * obs_ld[obs[0]][s]

    for t in range(1, T):
        for s in range(N):
            forward[s][t] = sum(forward[i][t - 1] * trans_prob[i][s] for i in range(N)) * obs_ld[obs[t]][s]
    
    return sum(row[-1] for row in forward)


def main():
    res = compute_forward(*parse_data())
    with open("result", "w") as f:
        f.write(str(res))


if __name__ == "__main__":
    main()

