import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from operator import add


def findDistances(k_start, k_end, dataSet):
    results = []
    for k in range(k_start, k_end):
        dist = euclidean_distances(dataSet[:, :k], dataSet[:, :k])
        dmax = dist.max()
        np.fill_diagonal(dist, np.inf)
        dmin = dist.min()
        r_of_k = np.log10((dmax - dmin) / dmin)
        results.append((k, r_of_k))

    return results


r_of_k = []
for n in [100, 1000]:
    mean_rk = [0] * 100
    for i in range(0, 10):
        dataSet = np.random.rand(n, n)
        k_vs_rk = findDistances(1, 100, dataSet)
        k, rk = zip(*k_vs_rk)
        mean_rk = list(map(add, mean_rk, rk))

    r_of_k.append(np.divide(mean_rk, 10))
    print("Done for {}".format(n))

fig, ax1 = plt.subplots()
color = 'red'
ax1.set_xlabel('k')
ax1.set_ylabel('r(k) n=100', color=color)
ax1.plot(range(1, 100), r_of_k[0], color=color)

ax2 = ax1.twinx()
color2 = 'blue'
ax2.set_ylabel('r(k) n=1000', color=color2)
ax2.plot(range(1, 100), r_of_k[1], color=color2)
fig.tight_layout()
plt.show()
