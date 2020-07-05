"""
This is created to support analyse performances of clustering implementation.
Created by zshuwei, 2020-06-11
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from math import exp, fabs


# methods
def calculate_sse(X, y):
    """
    X : array for clustering.

    y : array, shape = [n_samples]
         Predicted labels for each sample.
    Only calculate clusters labels >= 0
    """
    df = pd.DataFrame(X)
    df['y'] = y
    sse = 0
    for c in np.unique(y):
        if c < 0:
            continue
        cluster = df.loc[df['y'] == c]
        centroid = cluster.mean()[:-1].values
        for x in cluster.iloc[:, :-1].values:
            sse += np.sum((x - centroid) ** 2)
    return sse


def compare_plot(Xs, predictions):
    """
    Create a plot comparing multiple learners.
    `Xs` is a list of tuples containing:
        (title, x coord, y coord)

    `predictions` is a list of tuples containing
        (title, predicted classes)
    All the elements will be plotted against each other in a
    two-dimensional grid.
    """

    # We will use subplots to display the results in a grid
    nrows = len(Xs)
    ncols = len(predictions)

    fig = plt.figure(figsize=(16, 8))

    # Show each element in the plots returned from plt.subplots()
    for row, (row_label, X_x, X_y) in enumerate(Xs):
        for col, (col_label, y_pred) in enumerate(predictions):
            ax = plt.subplot(nrows, ncols, row * ncols + col + 1)
            if row == 0:
                plt.title(col_label)
            if col == 0:
                plt.ylabel(row_label)

            plt.scatter(X_x, X_y, c=y_pred.astype(np.float), cmap='prism', alpha=0.5)

            # Set the axis tick formatter to reduce the number of ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.show()
    plt.close()


def pca_var_plotbar(X, title=''):
    """
    Plot bar for PCA analysis on X
    """
    pca = PCA()
    principal_components = pca.fit_transform(X)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.title(title)
    plt.xticks(features)
    plt.show()


def pca_ratio_plotcurve(X, percentage=0.9, title=''):
    """
    Plot curve for PCA analysis on X
    """
    pca = PCA().fit(X)
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    xi = np.arange(1, len(pca.explained_variance_ratio_) + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 11, step=1))
    plt.ylabel('Cumulative variance (%)')
    plt.title(title)
    plt.axhline(y=percentage, color='r', linestyle='-')
    plt.text(0.5, percentage - 0.1, f"{percentage * 100}% cut-off threshold", color='red', fontsize=16)
    ax.grid(axis='x')
    plt.show()


def plot_sse(clusters, sse):
    plt.plot(clusters, sse, marker='o')
    plt.title("SSE trend on number of clusters")
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.show()


def plot_csm(X, y, n_cluster):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, y)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_cluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the y axis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(y.astype(float) / n_cluster)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    # calculate centers
    centers = []
    df = pd.DataFrame(X)
    df['y'] = y
    for c in range(n_cluster):
        cluster = df.loc[df['y'] == c]
        centers.append(cluster.mean()[:-1].values)
    centers = np.asarray(centers)

    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
    #               "with n_clusters = %d" % n_cluster),
    #              fontsize=14, fontweight='bold')
    plt.show()


def plot_dbscan(X, labels, core_sample_indices, eps, min_samples):
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title(f"DBSCAN with eps = {round(eps, 3)} and min_samples = {min_samples}")
    plt.show()


def clustering_rate(time, sse, csm, pca_components, k_cluster, noise_ratio=0):
    if noise_ratio == 1:
        return None
    rate_score = exp(time) * (sse * k_cluster / pca_components) * exp(-csm) / fabs(csm) / (1 - noise_ratio)
    return round(rate_score, 5)
