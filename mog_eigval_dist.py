# code from https://github.com/LMescheder/TheNumericsOfGANs/blob/master/notebooks/mog-eigval-dist.ipynb
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt


def kde(mu, tau, bbox=None, save_file="", xlabel="", ylabel="", cmap='Blues', show=False):
    if not bbox:
        bbox = [np.amin(mu, axis=0)-0.2, np.amax(mu,axis=0)+0.2, np.amin(tau,axis=0)-0.2, np.amax(tau,axis=0)+0.2]
    values = np.vstack([mu, tau])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='on',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='on')  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left='on',  # ticks along the bottom edge are off
        right='off',  # ticks along the top edge are off
        labelleft='on')  # labels along the bottom edge are off

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()


def complex_scatter(points, bbox=None, save_file="", xlabel="real part", ylabel="imaginary part", cmap='Blues'):
    fig, ax = plt.subplots()

    if bbox is not None:
        ax.axis(bbox)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx = [p.real for p in points]
    yy = [p.imag for p in points]

    plt.plot(xx, yy, 'X')
    plt.grid()

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_decision_boundary2d(model, bbox=[-2,2,-2,2], save_file=None):
    xx, yy = np.mgrid([bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j])
    """# X - some data in 2dimensional np.array

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# here "model" is your model's prediction (classification) function
Z = model(np.c_[xx.ravel(), yy.ravel()]) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=pl.cm.Paired)
plt.axis('off')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)"""