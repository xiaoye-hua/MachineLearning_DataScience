from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def kmeans_optimal_cluster(train_data: pd.DataFrame, cluster_nums: List[int]) -> None:
    """
    Find the optimal cluster number of Kmeans
    Args
        train_data:
        cluster_nums:

    Returns:

    """
    sc = []
    wcss = []
    for cluster_num in cluster_nums:
        model = KMeans(cluster_num)
        model.fit(train_data)
        res = model.predict(train_data)
        sc.append(silhouette_score(X=train_data, labels=res))
        wcss.append(model.inertia_)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplot(121)
    sns.lineplot(x=cluster_nums, y=sc)
    plt.xlabel("Cluster Number")
    plt.ylabel("Silhouette Score")

    plt.subplot(122)
    sns.lineplot(x=cluster_nums, y=wcss)
    plt.xlabel("Cluster Number")
    plt.ylabel("Within Cluster Sum of Square")
    plt.show()