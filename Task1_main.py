import time
from itertools import cycle, islice
from math import exp, fabs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler

pd.options.display.float_format = '{:,.3f}'.format


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


def clustering_rate(time, sse, csm, pca_components, k_cluster, noise_ratio=0):
    """
    Calculate clustering performance rate, the smaller the better
    """
    if noise_ratio == 1:
        return None
    rate_score = exp(time) * (sse * k_cluster / pca_components) * exp(-csm) / fabs(csm) / (1 - noise_ratio)
    return round(rate_score, 5)


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


if __name__ == '__main__':
    # #############################################################################
    # Read data from files
    dow_jones_filepath = "datasets\\dow_jones_index.data"
    live_filepath = "datasets\\Live.csv"
    sales_filepath = "datasets\\Sales_Transactions_Dataset_Weekly.csv"
    water_filepath = "datasets\\water-treatment.data"
    # #############################################################################
    print("\nReading data from file paths ...")
    dow_jones_data = pd.read_csv(
        dow_jones_filepath,
        encoding='utf-8',
        sep=',',
        skipinitialspace=True,
        header=0
    )
    live_data = pd.read_table(
        live_filepath,
        encoding='utf-8',
        sep=',',  # comma separated values
        skipinitialspace=True,
        na_values=['?'],
        index_col=None,
        header=0
    )
    sales_data = pd.read_table(
        sales_filepath,
        encoding='utf-8',
        sep=',',  # comma separated values
        skipinitialspace=True,
        index_col=0,
        header=0
    )

    water_data = pd.read_table(
        water_filepath,
        encoding='utf-8',
        sep=',',  # comma separated values
        skipinitialspace=True,
        na_values=['?'],
        index_col=None,
        header=None,
        names=["Date", "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E", "PH-P", "DBO-P", "SS-P",
               "SSV-P", "SED-P", "COND-P", "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D", "PH-S",
               "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S", "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S",
               "RD-DQO-S", "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G"]
    )
    # #############################################################################
    print("\nPre-processing dow_jones_index dataset ...")
    # Drop stock column
    dow_jones_data = dow_jones_data.drop(['stock'], axis=1)
    dow_jones_data = dow_jones_data.drop(['percent_change_price', 'percent_change_volume_over_last_wk',
                                          'days_to_next_dividend', 'percent_return_next_dividend'], axis=1)
    # Extract date (day) information from date
    dow_jones_data['date'] = pd.to_datetime(dow_jones_data['date'])
    dow_jones_data['day'] = dow_jones_data['date'].dt.day
    # Extract date (month) information from date
    dow_jones_data['month'] = dow_jones_data['date'].dt.month
    # Drop date column
    dow_jones_data = dow_jones_data.drop(['date'], axis=1)
    # Process with currency columns and convert into float type
    currency_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
    for c in currency_cols:
        dow_jones_data[c] = dow_jones_data[c].replace({'\$': '', ',': ''}, regex=True).astype(float)
    # Impute missing value
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(dow_jones_data.values)
    imputed_dow_jones_data = imr.transform(dow_jones_data.values)
    X_dow_jones = MinMaxScaler().fit_transform(imputed_dow_jones_data)

    print("\nPre-processing Facebook live dataset ...")
    # Drop empty columns
    live_data = live_data.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1)
    # Drop id column
    live_data = live_data.drop(['status_id'], axis=1)
    # drop label columns
    live_data = live_data.drop(['status_type'], axis=1)
    # Convert datetime to timestamp
    live_data['status_published'] = pd.to_datetime(live_data['status_published'])
    live_data['status_published'] = live_data.status_published.values.astype(np.int64) // 10 ** 9
    X_live = MinMaxScaler().fit_transform(live_data)

    print("\nPre-processing Sales Transactions dataset ...")
    # Drop MIN, MAX and Normalized columns
    normalised_cols = ['MIN', 'MAX']
    for col in sales_data.columns:
        if str(col).startswith('Normalize'):
            normalised_cols.append(str(col))
    sales_data = sales_data.drop(normalised_cols, axis=1)
    X_sales = MinMaxScaler().fit_transform(sales_data)

    print("\nPre-processing Water Treatment dataset ...")
    # Drop date column
    water_treatment_data = water_data.drop(['Date'], axis=1)
    # Impute missing value
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(water_treatment_data.values)
    imputed_water_treatment_data = imr.transform(water_treatment_data.values)
    X_water = MinMaxScaler().fit_transform(imputed_water_treatment_data)
    # #############################################################################
    pca_plots = [('Dow Jones Index', X_dow_jones),
                 ('Facebook Live', X_live),
                 ('Sales Transactions', X_sales),
                 ('Water Treatment', X_water)]
    # #############################################################################
    plot_pca = None
    if plot_pca:
        for dataset_name, dataset in pca_plots:
            pca_var_plotbar(dataset, title=dataset_name + ' PCA bar plot')
            pca_ratio_plotcurve(dataset, title=dataset_name + ' PCA curve plot')
    # #############################################################################
    # Base params for all datasets
    default_base = {'pca_components': 5,
                    'n_cluster': 2,
                    'noise_rate': 0,
                    'eps': 0.3,
                    'min_samples': 8,
                    'linkage': 'ward',
                    'random_state': 0}
    # Optimal params for datasets
    datasets = [(X_dow_jones, {'dataset_name': 'Dow Jones Index',
                               'pca_components': 3,
                               'km_cluster': 2,
                               'ac_cluster': 2,
                               'eps': .2,
                               'min_samples': 6,
                               'linkage': 'ward'}),
                (X_live, {'dataset_name': 'Facebook Live',
                          'pca_components': 3,
                          'km_cluster': 3,
                          'eps': .07,
                          'min_samples': 9,
                          'ac_cluster': 3,
                          'linkage': 'ward'}),
                (X_sales, {'dataset_name': 'Sales Transactions',
                           'pca_components': 5,
                           'km_cluster': 2,
                           'eps': .4,
                           'min_samples': 20,
                           'ac_cluster': 2,
                           'linkage': 'ward'}),
                (X_water, {'dataset_name': 'Water Treatment',
                           'pca_components': 2,
                           'km_cluster': 2,
                           'eps': .17,
                           'min_samples': 4,
                           'ac_cluster': 2,
                           'linkage': 'complete'})]
    # #############################################################################
    # Save results
    km_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))
    db_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))
    ac_df = pd.DataFrame(columns=('Dataset', 'Time', 'SSE', 'CSM'))
    # Plot diagrams
    plt.figure(figsize=(12, 21))
    plt.subplots_adjust(left=.05, right=.95, bottom=.001, top=.96, wspace=.3,
                        hspace=.3)
    plot_num = 1

    for i_dataset, (dataset, optimal_params) in enumerate(datasets):
        params = default_base.copy()
        params.update(optimal_params)
        # Feature selection
        pca = PCA(n_components=params['pca_components'])
        X = pca.fit_transform(dataset)
        # Create cluster objects
        km = KMeans(n_clusters=params['km_cluster'], random_state=params['random_state'])
        db = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        ac = AgglomerativeClustering(n_clusters=params['ac_cluster'], linkage=params['linkage'])
        clustering_algorithms = (('KMeans', km), ('DBSCAN', db), ('AgglomerativeClustering', ac))

        for name, algorithm in clustering_algorithms:
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            # Predict samples clusters
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
            # Metrics to rate clustering
            sample_silhouette_values = silhouette_samples(X, y_pred)
            running_time = t1 - t0
            sse = calculate_sse(X, y_pred)
            csm = silhouette_score(X, y_pred)
            n_pca = params['pca_components']
            n_cluster = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            noise_ratio = len([noise for noise in y_pred if noise == -1]) / len(y_pred)
            rate = clustering_rate(running_time, sse, csm, n_pca, n_cluster, noise_ratio)
            # Save results into DataFrame
            if name.startswith('K'):
                km_df = km_df.append({'Dataset': params['dataset_name'] + ' (' + str(params['km_cluster']) + ')',
                                      'Time': running_time,
                                      'SSE': sse,
                                      'CSM': csm},
                                     ignore_index=True)
            if name.startswith('DB'):
                db_df = db_df.append({'Dataset': params['dataset_name'],
                                      'Time': running_time,
                                      'SSE': sse,
                                      'CSM': csm},
                                     ignore_index=True)
            if name.startswith('Ag'):
                ac_df = ac_df.append({'Dataset': params['dataset_name'] + ' (' + params['linkage'] + ')',
                                      'Time': running_time,
                                      'SSE': sse,
                                      'CSM': csm},
                                     ignore_index=True)

            # ### Plot csm and scatter
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            # Plot CSM
            plt.subplot(8, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=16)

            ax1 = plt.gca()
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])
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
                y_lower = y_upper + 10

            # The vertical line for average silhouette score of all the values
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel(params['dataset_name'] + "\nCluster label")

            ax1.axvline(x=csm, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the y axis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.text(.99, .15, ('SSE: %.2f' % sse),
                     transform=plt.gca().transAxes, size=10,
                     horizontalalignment='right')
            plt.text(.99, .08, ('CSM: %.2f' % csm),
                     transform=plt.gca().transAxes, size=10,
                     horizontalalignment='right')
            plt.text(.99, .01, ('Rate: %.2f' % rate),
                     transform=plt.gca().transAxes, size=10,
                     horizontalalignment='right')

            plt.subplot(8, len(clustering_algorithms), plot_num + len(clustering_algorithms))

            ax2 = plt.gca()
            ax2.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
            # Calculate centroids
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

            for center_k, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % center_k, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("Clustered data %.3fs" % running_time)
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plot_num += 1
        plot_num += len(clustering_algorithms)
    plt.show()

    # Print results
    print('Best K Means results:\n', km_df)
    print('Best DBSCAN results:\n', db_df)
    print('Best Agglomerative results:\n', ac_df)
