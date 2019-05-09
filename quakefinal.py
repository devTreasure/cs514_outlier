import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
# centers = [[1, 1], [-1, -1], [1, -1]]
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import sys
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

df = pd.read_csv('D:\\quake.dat', header=None, skiprows=7)
# print(df)


ss = StandardScaler()
X = ss.fit_transform(df)

# Train the isolation forest
isf = IsolationForest(n_estimators=150, behaviour='new', contamination=0.01, random_state=1000)
Y_pred = isf.fit_predict(X)
Yf= Y_pred


print('Outliers  using isolation forest  set: {}'.format(np.sum(Y_pred == -1)))
y_pred= np.array(X)



fig, ax = plt.subplots(figsize=(15, 10))


ax.scatter(X[Yf > 0, 0], X[Yf > 0, 1], marker='^', s=80, label='inliers')
ax.scatter(X[Yf == -1, 0], X[Yf == -1, 1], marker='x',c= 'black', s=50, label='Ouliers')


ax.set_xlabel(r'features', fontsize=14)
ax.set_ylabel(r'features', fontsize=14)

ax.legend(fontsize=14)

plt.show()



nb_clusters = [4, 6, 8, 10]
linkages = ['single', 'complete', 'ward', 'average']
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method Kmeans optimization')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

ss = StandardScaler(with_std=False)
sdf = ss.fit_transform(df)

# Perform the TSNE non-linear dimensionality reduction
tsne = TSNE(n_components=2, perplexity=10, random_state=1000)
data_tsne = tsne.fit_transform(sdf)

df_tsne = pd.DataFrame(data_tsne, columns=['x', 'y'], index=df.index)
dff = pd.concat([df, df_tsne], axis=1)

# Show the dataset
sns.set()

fig, ax = plt.subplots(figsize=(18, 11))

with sns.plotting_context("notebook", font_scale=1.5):
    sns.scatterplot(x='x',
                    y='y',
                    size=0,
                    sizes=(120, 120),
                    data=dff,
                    legend=False,
                    ax=ax)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

plt.show()

# Analyze the result of different linkages and number of clusters
cpcs = np.zeros(shape=(len(linkages), len(nb_clusters)))
silhouette_scores = np.zeros(shape=(len(linkages), len(nb_clusters)))

for i, l in enumerate(linkages):
    for j, nbc in enumerate(nb_clusters):
        dm = pdist(sdf, metric='minkowski', p=2)
        Z = linkage(dm, method=l)
        cpc, _ = cophenet(Z, dm)
        cpcs[i, j] = cpc

        ag = AgglomerativeClustering(n_clusters=nbc, affinity='euclidean', linkage=l)
        Y_pred = ag.fit_predict(sdf)
        sls = silhouette_score(sdf, Y_pred, random_state=1000)
        silhouette_scores[i, j] = sls

fig, ax = plt.subplots(len(nb_clusters), 2, figsize=(20, 20), sharex=True)

for i in range(len(nb_clusters)):
    ax[i, 0].plot(cpcs[:, i])
    #ax[i, 0].set_ylabel('Cophenetic correlation', fontsize=14)
    ax[i, 0].set_title('Number of clusters: {}'.format(nb_clusters[i]), fontsize=14)

    ax[i, 1].plot(silhouette_scores[:, i])
    ax[i, 1].set_ylabel('Silhouette score', fontsize=14)
    ax[i, 1].set_title('Number of clusters: {}'.format(nb_clusters[i]), fontsize=14)

plt.xticks(np.arange(len(linkages)), linkages)

plt.show()

# Show the truncated dendrogram for a complete linkage
dm = pdist(sdf, metric='euclidean')
Z = linkage(dm, method='complete')

fig, ax = plt.subplots(figsize=(25, 20))

#d = dendrogram(Z, orientation='right', truncate_mode='lastp', p=80, no_labels=True, ax=ax)
#ax.set_xlabel('Dissimilarity', fontsize=18)
#ax.set_ylabel('Samples (80 leaves)', fontsize=18)
#plt.show()

# Perform the clustering
for n in (3, 4):
    ag = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='complete')
    Y_pred = ag.fit_predict(sdf)

    df_pred = pd.Series(Y_pred, name='Cluster', index=df.index)
    pdff = pd.concat([dff, df_pred], axis=1)

    # Show the results of the clustering
    fig, ax = plt.subplots(figsize=(18, 11))

    with sns.plotting_context("notebook", font_scale=1.5):
        sns.scatterplot(x='x',
                        y='y',
                        hue='Cluster',
                        size='Cluster',
                        sizes=(120, 120),
                        palette=sns.color_palette("husl", n),
                        data=pdff,
                        ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.show()

