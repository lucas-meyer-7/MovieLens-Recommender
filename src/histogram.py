import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data: np.ndarray, ax: plt.Axes, k: int) -> None:
    """
    Plot a histogram of the given data on the given axis.

    Args:
     - data (ndarray): the data to plot.
     - ax (plt.Axes) : the axis to plot the histogram on.
     - k (int): the value of k used to generate the data.

    Returns: None
    """
    nbins = 100
    hist, bin_spec = np.histogram(data, nbins)
    a, b = min(bin_spec), max(bin_spec)
    dx = (b-a)/nbins
    ax.bar(bin_spec[:-1]+dx/2.0, hist, width=dx)
    ax.set_title(f"$k = {k}$ and $\\lambda = 0.1$")
    ax.set_xlabel("Predicted rating ($\\mathbf{u}^T \\mathbf{v}$)")
    ax.set_ylabel("Frequency")


fig, axs = plt.subplots(2, 2, figsize=(10, 4))
fig.suptitle("Four histograms of the dot products of 1000 randomly"
             + "initialized user and item vectors", fontsize=13)
fig.subplots_adjust(hspace=1, top=0.8)

# Use different values of k
for i, k in enumerate([1, 2, 5, 20]):
    mean = (np.sqrt(2.5/k))
    var = 0.1
    std = np.sqrt(var)
    print(f"Mean = {mean}, Standard Deviation = {std}")

    sample = []
    for j in range(1000):
        u = np.random.normal(loc=mean, scale=std, size=k)
        v = np.random.normal(loc=mean, scale=std, size=k)
        dp = np.inner(u, v)
        sample.append(dp)
    plot_histogram(sample, axs[i//2, i % 2], k)

plt.savefig("../report/graphics/histograms.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
