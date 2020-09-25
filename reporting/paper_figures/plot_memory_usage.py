import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def usual_size(m):
    return 4*m**2

def pushfwd_size(m, n, dc):
    return 4*n*m*dc

def invop_size(n, dc):
    return 4*n*dc**2

def implicit_size(m, n, dc):
    return pushfwd_size(m, n, dc) + invop_size(n, dc)

data_size = 1000

ms = np.linspace(1000, 1e6, 100)
plt.semilogy(ms, 1e-9 * usual_size(ms), label="explicit")
plt.semilogy(ms, 1e-9 * implicit_size(ms, n=1000, dc=1), label="implicit")
plt.legend()
plt.ylabel("memory footprint [GB]")
plt.xlabel("model size")
plt.show()
plt.savefig("memory_footprint_explicit_vs_implicit", bbox_inches="tight", pad_inches=0, dpi=600)

"""
plt.plot(ms, 1e-9 * implicit_size(ms, n=1000, dc=1), label="implicit, chunk size = 1")
plt.plot(ms, 1e-9 * implicit_size(ms, n=100, dc=10), label="implicit, chunk size = 10")
plt.plot(ms, 1e-9 * implicit_size(ms, n=10, dc=100), label="implicit, chunk size = 100")
plt.plot(ms, 1e-9 * implicit_size(ms, n=1, dc=1000), label="implicit, chunk size = 1000")
plt.legend()
plt.ylabel("memory footprint [GB]")
plt.xlabel("model size")
plt.show()
"""
