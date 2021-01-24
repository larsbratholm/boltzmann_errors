import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.neighbors
import sklearn.model_selection
import seaborn as sns
from matplotlib import rc

# Set fig size
plt.rcParams["figure.figsize"] = (16,9)
##Set plotting theme
sns.set(font_scale=2.,rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)
# Set "b", "g", "r" to default seaborn colors
sns.set_color_codes("deep")



def sample(energies, var, corr, n_samples=10000):
    """ Sample boltzmann weights, given energies (in units of kbT),
        variance and correlation.
    """
    n = len(energies)

    cov = np.identity(n) * var + (np.ones((n,n)) - np.identity(n)) * var * corr

    np.random.seed(42)
    sampled_energies = scipy.stats.multivariate_normal.rvs(mean=energies, cov=cov, size=n_samples)

    p = np.exp(-sampled_energies)
    p /= p.sum(1)[:,None]
    return p

def plot_distribution(p, energies):
    """ Plots the distribution using KDE's on the ĺogit-transform
    """
    def get_probability(logp, grid):
        kde = sklearn.neighbors.KernelDensity()
        cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True)
        param_grid = {'bandwidth': 10**np.linspace(-2, 0, 100)}
        cv_estimator = sklearn.model_selection.GridSearchCV(kde, param_grid, cv=cv)

        cv_estimator.fit(logp[:,None])

        return np.exp(cv_estimator.best_estimator_.score_samples(grid[:,None]).ravel())

    def logit(x):
        return np.log(x/(1-x))

    def reverse_logit(y):
        z = np.exp(y)
        return z / (1 + z)

    # old fashioned exact boltzmann
    energies = np.asarray(energies)
    standard_p = np.exp(-energies)
    standard_p /= standard_p.sum()

    transform = logit(p)
    range_ = transform.max() - transform.min()
    grid = np.linspace(transform.min() - range_/20, transform.max() + range_/20, 10000)

    probabilities = []

    x = reverse_logit(grid)
    ymax = 0
    for i in range(p.shape[1]):
        y = get_probability(transform[:,i], grid)
        ymax = max(ymax, y.max())
        plt.plot(x, y)
        c = plt.gca().lines[-1].get_color()
        plt.fill_between(x, 0, y, alpha=0.15, color=c)
        plt.vlines(standard_p[i], 0, 1.3, color=c)
        plt.vlines(np.mean(p, axis=0), 0, 1.3, color=c, linestyle="dashed")
    plt.xlim([0,1])
    plt.ylim([0, ymax*1.05])
    plt.xlabel("Boltzmann weight")
    plt.ylabel("Density")
    plt.savefig("4.png", pad_inches=0.0, bbox_inches="tight", dpi=300)





if __name__ == "__main__":
    energies = [0,1,2]
    p = sample(energies, 4, 0.75, n_samples=1000)
    plot_distribution(p, energies)
