import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

LAMBDAS = [0., 0.1, 0.4, 0.6, 0.9, 1.]


def anisotropic_score(a, b, lambda_=0.5):
    # dot product
    cos = np.einsum('dn, dc -> nc', a / np.linalg.norm(a, axis=0, keepdims=True),
                    b / np.linalg.norm(b, axis=0, keepdims=True))  # collapse empty dimensions
    # lengths
    abs = np.abs(np.linalg.norm(a[..., None], axis=0) - np.linalg.norm(b[:, None, :],
                                                                       axis=0))  # note: abs is somewhat arbitrary here, could also use mse, ...
    # both have shape N, C
    return lambda_ * (-1. * cos) + (1. - lambda_) * abs  # note: -1 * cos for HIGHEST similarity


if __name__ == "__main__":
    N = 2000
    C = 10
    a = np.random.randn(2, N)
    print(f"running with {N} samples and {C} catalog entries")

    top_k = N // 10  # we pick the top-k elements according to the anisotropic metric (for visualization)

    catalog = np.random.randn(2, C)  # the catalog
    colors = np.random.randint(0, 255, size=(C, 3)) / 255.

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))

    ix = 0
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            lambda_ = LAMBDAS[ix]

            ax[i, j].scatter(catalog[0, :], catalog[1, :], s=200, marker="o", label="Catalog Entries", c=colors,
                             edgecolor="black")

            ax[i, j].scatter(a[0, :], a[1, :], label="data", alpha=0.2, s=10)

            ax[i, j].grid()
            ax[i, j].legend()

            score = anisotropic_score(a, catalog, lambda_=lambda_)
            # for each C, pick the top_k best examples
            score = rearrange(score, 'n c -> c n')
            neighbor_ix = np.argsort(score, axis=1)[:, :top_k]
            neighbor_ix = rearrange(neighbor_ix, 'c k -> k c')

            neighbors = a[:, neighbor_ix]

            for n in range(C):
                ax[i, j].scatter(neighbors[0, :, n], neighbors[1, :, n], c=colors[n], s=10, marker="x")

            ax[i, j].set_title(f"lambda={lambda_:.2f}")
            ix += 1
    plt.savefig(f"plots/anisotropic_full.png")
    plt.show()

