import time
from itertools import cycle, islice
from math import exp, fabs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler

pd.options.display.float_format = '{:,.3f}'.format


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


def clustering_rate(time, sse, csm, pca_components, k_cluster, noise_ratio=0):
    if noise_ratio == 1:
        return None
    rate_score = exp(time) * (sse * k_cluster / pca_components) * exp(-csm) / fabs(csm) / (1 - noise_ratio)
    return round(rate_score, 5)


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


if __name__ == '__main__':
    filepath = 'datasets//Sales_Transactions_Dataset_Weekly.csv'
    print("\nReading data from {}".format(filepath))
    sales_data = pd.read_table(
        filepath,
        encoding='utf-8',
        sep=',',  # comma separated values
        skipinitialspace=True,
        index_col=0,
        header=0
    )

    print("\nStep 1: Pre-processing data ... ...")
    normalised_cols = ['MIN', 'MAX']
    for col in sales_data.columns:
        if str(col).startswith('Normalize'):
            normalised_cols.append(str(col))

    sales_data = sales_data.drop(normalised_cols, axis=1)

    print("\nStep 2: Normalise data ... ...")
    X_sales = MinMaxScaler().fit_transform(sales_data)

    # pca_var_plotbar(X_sales)
    pca_ratio_plotcurve(X_sales)

    D_PCA = 5
    print("\tChoose %d" % D_PCA, "as PCA components number")
    pca = PCA(n_components=D_PCA)
    X_r_sales = pca.fit_transform(X_sales)

    # #############################################################################
    default_base = {'clusters_range': range(2, 6),
                    'eps': [0.35, 0.40],
                    'min_samples': [15, 20],
                    'linkages': ['single', 'complete', 'ward', 'average']}

    datasets = [(X_r_sales, {})]

    # #############################################################################
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.2,
                        hspace=.2)
    plot_num = 1

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        params = default_base.copy()
        params.update(algo_params)

        X = dataset
        print('\nStep 3: Evaluating K Means Clustering ...')
        for n_cluster in params['clusters_range']:
            km = KMeans(n_clusters=n_cluster,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
            t0 = time.time()
            km.fit(X)
            t1 = time.time()

            if hasattr(km, 'labels_'):
                y_pred = km.labels_.astype(np.int)
            else:
                y_pred = km.predict(X)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])

            plt.subplot(2, len(params['clusters_range']), plot_num)
            ax1 = plt.gca()
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
            silhouette_avg = silhouette_score(X, y_pred)
            sample_silhouette_values = silhouette_samples(X, y_pred)
            y_lower = 10
            for k in range(n_cluster):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[y_pred == k]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=colors[k], edgecolor=colors[k], alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # The vertical line for average silhouette score of all the values
            ax1.set_title('K Means: %d' % n_cluster)
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the y axis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.text(.99, .04, ('CSM: %.2f' % silhouette_avg),
                     transform=plt.gca().transAxes, size=10,
                     horizontalalignment='right')
            plt.text(.99, .01,
                     ('Rate: %.2f' % clustering_rate(t1 - t0, km.inertia_, silhouette_avg, D_PCA, n_cluster, 0)),
                     transform=plt.gca().transAxes, size=10,
                     horizontalalignment='right')

            plt.subplot(2, len(params['clusters_range']), plot_num + len(params['clusters_range']))
            ax2 = plt.gca()

            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors[y_pred], edgecolor='k')

            centers = []
            df = pd.DataFrame(X)
            df['y'] = y_pred
            for c in range(n_cluster):
                cluster = df.loc[df['y'] == c]
                centers.append(cluster.mean()[:-1].values)
            centers = np.asarray(centers)

            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for k, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % k, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("Clustered data %.2fs" % (t1 - t0))
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plot_num += 1

        plt.show()

        # #############################################################################
        print('\nStep 4: Evaluating DBSCAN Clustering ...')

        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.2,
                            hspace=.2)
        X = X_r_sales
        plot_num = 1
        params = default_base.copy()
        for i in range(len(params['eps'])):
            for j in range(len(params['min_samples'])):
                e = params['eps'][i]
                ms = params['min_samples'][j]
                db = DBSCAN(eps=e, min_samples=ms)
                t0 = time.time()
                db.fit(X)
                t1 = time.time()

                if hasattr(db, 'labels_'):
                    y_pred = db.labels_.astype(np.int)
                else:
                    y_pred = db.predict(X)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])

                n_cluster = len(set(y_pred)) - (1 if -1 in y_pred else 0)

                noise_rate = len([noise for noise in y_pred if noise == -1]) / len(y_pred)

                subplot_num = len(params['eps']) * len(params['min_samples'])

                plt.subplot(2, subplot_num, plot_num)
                ax1 = plt.gca()
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
                sse_db = calculate_sse(X, y_pred)
                silhouette_avg = silhouette_score(X, y_pred)
                sample_silhouette_values = silhouette_samples(X, y_pred)
                y_lower = 10
                for k in range(n_cluster):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[y_pred == k]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=colors[k], edgecolor=colors[k], alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                # The vertical line for average silhouette score of all the values
                ax1.set_title(f'DBSCAN(eps={e}, min_samples={ms}: n_cluster={n_cluster}')
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                ax1.set_yticks([])  # Clear the y axis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                plt.text(.99, .04, ('CSM: %.2f' % silhouette_avg),
                         transform=plt.gca().transAxes, size=10,
                         horizontalalignment='right')
                plt.text(.99, .01, ('Rate: %.2f' % clustering_rate(t1 - t0, sse_db, silhouette_avg,
                                                                   D_PCA, n_cluster, noise_rate)),
                         transform=plt.gca().transAxes, size=10,
                         horizontalalignment='right')

                plt.subplot(2, subplot_num, plot_num + subplot_num)
                ax2 = plt.gca()

                ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors[y_pred], edgecolor='k')

                centers = []
                df = pd.DataFrame(X)
                df['y'] = y_pred
                for c in range(n_cluster):
                    cluster = df.loc[df['y'] == c]
                    centers.append(cluster.mean()[:-1].values)
                centers = np.asarray(centers)

                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for k, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % k, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("Clustered data %.2fs" % (t1 - t0))
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plot_num += 1

        plt.show()

        # #############################################################################
        print('\nStep 5: Evaluating Agglomerative Clustering ...')
        for lkg in params['linkages']:

            plt.figure(figsize=(20, 10))
            plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.2,
                                hspace=.2)
            X = X_r_sales
            plot_num = 1
            params = default_base.copy()

            for n_cluster in params['clusters_range']:
                ac = AgglomerativeClustering(n_clusters=n_cluster, linkage=lkg)
                t0 = time.time()
                ac.fit(X)
                t1 = time.time()

                if hasattr(ac, 'labels_'):
                    y_pred = ac.labels_.astype(np.int)
                else:
                    y_pred = ac.predict(X)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])

                n_cluster = len(set(y_pred)) - (1 if -1 in y_pred else 0)

                noise_rate = 0

                subplot_num = len(params['clusters_range'])

                plt.subplot(2, subplot_num, plot_num)
                ax1 = plt.gca()
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
                sse_db = calculate_sse(X, y_pred)
                silhouette_avg = silhouette_score(X, y_pred)
                sample_silhouette_values = silhouette_samples(X, y_pred)
                y_lower = 10
                for k in range(n_cluster):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[y_pred == k]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=colors[k], edgecolor=colors[k], alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                # The vertical line for average silhouette score of all the values
                ax1.set_title(f'Agglomerative({lkg}): {n_cluster}')
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                ax1.set_yticks([])  # Clear the y axis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                plt.text(.99, .04, ('CSM: %.2f' % silhouette_avg),
                         transform=plt.gca().transAxes, size=10,
                         horizontalalignment='right')
                plt.text(.99, .01, ('Rate: %.2f' % clustering_rate(t1 - t0, sse_db, silhouette_avg,
                                                                   D_PCA, n_cluster, noise_rate)),
                         transform=plt.gca().transAxes, size=10,
                         horizontalalignment='right')

                plt.subplot(2, subplot_num, plot_num + subplot_num)
                ax2 = plt.gca()

                ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors[y_pred], edgecolor='k')

                centers = []
                df = pd.DataFrame(X)
                df['y'] = y_pred
                for c in range(n_cluster):
                    cluster = df.loc[df['y'] == c]
                    centers.append(cluster.mean()[:-1].values)
                centers = np.asarray(centers)

                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for k, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % k, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("Clustered data %.2fs" % (t1 - t0))
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plot_num += 1

            plt.show()
