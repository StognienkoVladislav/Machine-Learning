import pandas as pd
import numpy as np

from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


iris = datasets.load_iris()

iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

# let`s remove spaces from column name
iris.columns = iris.columns.str.replace(" ", "")
iris.head()

X = iris.ix[:, :3]      # independent variables
y = iris.species        # dependent variable
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Agglomerative Cluster
model = AgglomerativeClustering(n_clusters=3)
model.fit(X)

iris['pred_species'] = model.labels_

print("Accuracy :", metrics.accuracy_score(iris.species, iris.pred_species))
print("Classification report :", metrics.classification_report(iris.species, iris.pred_species))

# generate the linkage matrix
Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Agglomerative Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.tight_layout()
plt.show()
