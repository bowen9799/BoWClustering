from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)



# IRIS DATA
# iris = sklearn.datasets.load_iris()
# X = iris.data[:, :4]  # extract only the features
X = np.array([[1],[2],[3],[4],[10],[11],[12],[13],[100],[200],[300],[400]])
# X = np.array([[1.        ] ,
#  [0.852743  ] ,
#  [0.8683993 ] ,
#  [0.8071629 ] ,
#  [0.25270075] ,
#  [0.24268186] ,
#  [0.2193648 ] ,
#  [0.24373257] ,
#  [0.774263  ] ,
#  [0.7715119 ] ,
#  [0.7375568 ] ,
#  [0.7539828 ] ,
#  [0.22433582] ,
#  [0.1845077 ] ,
#  [0.22274964] ,
#  [0.23183177] ,
#  [0.35953194] ,
#  [0.5442772 ] ,
#  [0.42991602] ,
#  [0.44122118] ,
#  [0.09524228] ,
#  [0.07642484] ,
#  [0.07747524] ,
#  [0.06596614] ,
#  [0.32441327] ,
#  [0.29882973] ,
#  [0.34657937] ,
#  [0.47760355] ,
#  [0.605088  ] ,
#  [0.66972136] ,
#  [0.68631077] ,
#  [0.545611  ] ,
#  [0.55845535] ,
#  [0.7208049 ] ,
#  [0.647128  ] ,
#  [0.2501768 ] ,
#  [0.32502922] ,
#  [0.30983013] ,
#  [0.621074  ] ,
#  [0.6091449 ] ,
#  [0.58516914] ,
#  [0.6043534 ] ,
#  [0.16855446] ,
#  [0.12579826] ,
#  [0.1427682 ] ,
#  [0.14667119] ,
#  [0.30902648] ,
#  [0.34107825] ,
#  [0.23477808] ,
#  [0.28918755] ,
#  [0.05041694] ,
#  [0.04612006] ,
#  [0.05687741] ,
#  [0.07294004]])
# for n in X:
#     n[0] = n[0] * 1000000
# print X
# Xs = StandardScaler().fit_transform(X)
# Y = iris.target

ks = range(1,10)

# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]

# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

# print BIC