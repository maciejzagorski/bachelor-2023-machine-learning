import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import rand_score
from sklearn.metrics.cluster import contingency_matrix

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

data_uncompressed = pd.read_csv('../data/report_2_clustering_algorithms_iris_org.csv')

print(data_uncompressed)

stdsc = StandardScaler()
mms = MinMaxScaler()
pca = PCA(n_components=2)

data_standardized_uncompressed = stdsc.fit_transform(data_uncompressed.iloc[:, :-1])
data_standardized_compressed = pca.fit_transform(data_standardized_uncompressed)

data_standardized_compressed = pd.DataFrame(data_standardized_compressed,
                                            columns=['"PC1"', '"PC2"'])

print(data_standardized_compressed)

data_iris = pd.read_csv('../data/report_2_clustering_algorithms_iris_2d.csv')

data_iris.drop([data_iris.columns[0]], axis=1, inplace=True)

X = np.array(data_iris)
X_norm = mms.fit_transform(data_iris)
X_stand = stdsc.fit_transform(data_iris)

X_data_sets = {
    "Raw data": X,
    "Normalized": X_norm,
    "Standardized": X_stand,
}


def plot_data(data_set, axis, data_set_name):
    axis.scatter(data_set[:, 0], data_set[:, 1], c='white', marker='o',
                 edgecolor='black', s=50)
    axis.set_title(data_set_name, style="italic")
    axis.grid()


def plot_data_loop(data_sets, plot_title):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    axes = [ax1, ax2, ax3]
    i = 0

    for data_set_name, data_set in data_sets.items():
        plot_data(data_set, axes[i], data_set_name)
        i += 1

    f.suptitle(plot_title, weight='bold')
    plt.tight_layout()
    f.savefig('../plots/report_2_clustering_algorithms_plot_' + plot_title + '_300.svg', format='svg', dpi=300)


plot_data_loop(X_data_sets, "Data visualisation")

colors = ['red', 'green', 'orange', 'blue', 'yellow', 'indigo', 'violet', 'cyan',
          'sienna', 'teal', 'purple', 'olive', 'lime', 'crimson', 'aqua', 'pink']
markers = ['^', 's', 'p', '>', 'o', 'h', 'v', 'd', '*', '<', '8', 'P', 'H', 'X', 'D']


def plot_fit_data(data_set, axis, data_set_name, algorithm, results):
    if algorithm == 'km':
        y = km.fit_predict(data_set)
    elif algorithm == 'db':
        y = db.fit_predict(data_set)
    else:
        y = ac.fit_predict(data_set)

    for i in range(min(y), max(y) + 1):
        if i == -1:
            lb = "Noise"
        else:
            lb = f'Cluster {i + 1}'

        axis.scatter(data_set[y == i, 0],
                     data_set[y == i, 1],
                     s=50, c=colors[i],
                     marker=markers[i], edgecolor='black',
                     label=lb)

    if algorithm == 'km':
        axis.scatter(km.cluster_centers_[:, 0],
                     km.cluster_centers_[:, 1],
                     s=250, marker='*',
                     c='grey', edgecolor='black',
                     label='Centroids')

    axis.set_title(data_set_name, style="italic")
    axis.legend(scatterpoints=1)
    axis.grid()

    results.append(y)


def plot_fit_data_loop(data_sets, plot_title, algorithm):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    axes = [ax1, ax2, ax3]
    i = 0
    results = []

    for data_set_name, data_set in data_sets.items():
        plot_fit_data(data_set, axes[i], data_set_name, algorithm, results)
        i += 1

    f.suptitle(plot_title, weight='bold')
    plt.tight_layout()
    f.savefig('../plots/report_2_clustering_algorithms_plot_' + plot_title + '_300.svg', format='svg', dpi=300)

    return results


km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04,
            random_state=0)
km_results = plot_fit_data_loop(X_data_sets, "k-means, init='random'", "km")

km.set_params(init='k-means++')
km_pp_results = plot_fit_data_loop(X_data_sets, "k-means, init='k-means++'", "km")

db = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
db_results = plot_fit_data_loop(X_data_sets, "DBSCAN, eps=0.5, min_samples=5", "db")

db.set_params(eps=0.3195)
db_eps_results = plot_fit_data_loop(X_data_sets, "DBSCAN, eps=0.3195, min_samples=5",
                                    "db")

db.set_params(eps=0.3195, min_samples=7)
db_eps_min_samp_results = plot_fit_data_loop(X_data_sets, "DBSCAN, eps=0.3195, "
                                                          "min_samples=7", "db")

ac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
ac_results = plot_fit_data_loop(X_data_sets, "Agglomerative Clustering, "
                                             "linkage='complete'", "ac")

plt.show()


def purity_score(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def score_loop(data_sets, score_title, labels_true, results):
    i = 0
    for data_set_name in data_sets:
        print(score_title + ": " + data_set_name + ": Purity [%%]: %.2f" %
              (purity_score(labels_true, results[i]) * 100))
        print(score_title + ": " + data_set_name + ": Rand index [%%]: %.2f" %
              (rand_score(labels_true, results[i]) * 100))
        print()
        i += 1


data_iris_results = pd.read_csv('../data/report_2_clustering_algorithms_iris_org.csv')

y_true = data_iris_results.iloc[:, -1]

score_loop(X_data_sets, "K-means, init='random'", y_true, km_results)
score_loop(X_data_sets, "K-means, init='k-means++'", y_true, km_pp_results)
score_loop(X_data_sets, "DBSCAN, eps=0.5, min_samples=5", y_true, db_results)
score_loop(X_data_sets, "DBSCAN, eps=0.3195, min_samples=5", y_true, db_eps_results)
score_loop(X_data_sets, "DBSCAN, eps=0.3195, min_samples=7", y_true,
           db_eps_min_samp_results)
score_loop(X_data_sets, "Agglomerative Clustering, linkage='complete'", y_true,
           ac_results)

f, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
km.set_params(init='random')
y = km.fit_predict(X)

for i in range(min(y), max(y) + 1):
    if i == -1:
        lb = "Noise"
    else:
        lb = f'Cluster {i + 1}'

    ax1.scatter(X[y == i, 0],
                X[y == i, 1],
                s=50, c=colors[i],
                marker=markers[i], edgecolor='black',
                label=lb)

ax1.set_title("Raw data", style="italic")
ax1.legend(scatterpoints=1)
ax1.grid()

f.suptitle("K-means, init='random'", weight='bold')
plt.tight_layout()
f.savefig('../plots/report_2_clustering_algorithms_plot_k_means_random_1_300.svg', format='svg', dpi=300)

plt.show()

f, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
y = ac.fit_predict(X)

for i in range(min(y), max(y) + 1):
    if i == -1:
        lb = "Noise"
    else:
        lb = f'Cluster {i + 1}'

    ax1.scatter(X[y == i, 0],
                X[y == i, 1],
                s=50, c=colors[i],
                marker=markers[i], edgecolor='black',
                label=lb)

ax1.set_title("Raw data", style="italic")
ax1.legend(scatterpoints=1)
ax1.grid()

f.suptitle("Agglomerative Clustering, linkage='complete'", weight='bold')
plt.tight_layout()
f.savefig('../plots/report_2_clustering_algorithms_plot_ac_1_300.svg', format='svg', dpi=300)

plt.show()

np_arr = np.array(y_true)

np_arr[np_arr=='Setosa'] = 0
np_arr[np_arr=='Versicolor'] = 1
np_arr[np_arr=='Virginica'] = 2

f, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
y = np_arr

for i in range(min(y), max(y) + 1):
    if i == -1:
        lb = "Noise"
    else:
        lb = f'Cluster {i + 1}'

    ax1.scatter(X[y == i, 0],
                X[y == i, 1],
                s=50, c=colors[i],
                marker='o', edgecolor='black',
                label=lb)

ax1.set_title("Raw data", style="italic")
ax1.legend(scatterpoints=1)
ax1.grid()

f.suptitle("Actual clustering", weight='bold')
plt.tight_layout()
f.savefig('../plots/report_2_clustering_algorithms_plot_true_300.svg', format='svg', dpi=300)

plt.show()