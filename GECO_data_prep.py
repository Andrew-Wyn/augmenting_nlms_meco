import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min


seed_ = 12345
np.random.seed(seed_)


def clusterize_user_profiling(reader_grouped_df, gaze_features):
    """
    Apply the K-Means algorithm to retrieve K clusters and the relative 
    """

    scaler = MinMaxScaler()

    X = scaler.fit_transform(reader_grouped_df[gaze_features].values)

    sse_list = list()
    separations = list()
    silouettes_ = list()

    max_k = 10
    for k in tqdm(range(2, max_k + 1)):
        kmeans = KMeans(n_clusters=k, random_state=seed_, n_init=100, max_iter=100)
        kmeans.fit(X)

        sse = kmeans.inertia_
        sse_list.append(sse)
        separations.append(metrics.davies_bouldin_score(X, kmeans.labels_))
        silouettes_.append(silhouette_score(X, kmeans.labels_))


    selected_k=5

    kmeans = KMeans(n_clusters=selected_k, random_state=seed_, n_init=100, max_iter=500)
    kmeans.fit(X)

    # sum up the metrics

    print(f"SSE : {kmeans.inertia_}")
    print(f"Separation : {metrics.davies_bouldin_score(X, kmeans.labels_)}")
    print(f"Silhouette : {silhouette_score(X, kmeans.labels_)}")

    center = scaler.inverse_transform(kmeans.cluster_centers_)

    # TODO: return the nearest user to each cluster
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    # clostest[i] -> contains the index of the point closest to the i-th centroid
    medoids = []
    
    for i in range(selected_k):
        medoids.append(reader_grouped_df.PP_NR[closest[i]])
        
    return medoids


def main():
    df = pd.read_csv("augmenting_nlms_meco_data/geco_monolingual.csv", index_col=0)

    gaze_features = ["WORD_FIRST_FIXATION_DURATION", "WORD_FIRST_RUN_END_TIME", "WORD_FIRST_RUN_START_TIME", "WORD_GAZE_DURATION", "WORD_FIRST_RUN_FIXATION_COUNT", "WORD_FIXATION_COUNT"]
    other_features = ["PP_NR", "PART", "TRIAL", "WORD_ID", "WORD"] # trial is the paragraph readed
    df = df[other_features + gaze_features]

    # Reasonably the features with "." are related to skipped words
    # We can substitute them with 0 value.
    for gaze_feat in gaze_features:
        df.loc[df[gaze_feat] == ".", gaze_feat] = 0

    df[gaze_features] = df[gaze_features].astype(float)

    df["WORD_FIRST_RUN_DURATION"] = df["WORD_FIRST_RUN_END_TIME"] - df["WORD_FIRST_RUN_START_TIME"]

    del df["WORD_FIRST_RUN_END_TIME"]
    del df["WORD_FIRST_RUN_START_TIME"]

    gaze_features.remove("WORD_FIRST_RUN_END_TIME")
    gaze_features.remove("WORD_FIRST_RUN_START_TIME")

    mapping_columns = {
        "PP_NR": "uniform_id",
        "PART": "trialid",
        "TRIAL": "sentnum",
        "WORD_ID": "ianum",
        "WORD": "ia",
        "PP_NR": "uniform_id",
        "WORD_FIRST_FIXATION_DURATION": "firstfix_dur",
        "WORD_FIRST_RUN_DURATION": "firstrun_dur",
        "WORD_GAZE_DURATION": "dur",
        "WORD_FIRST_RUN_FIXATION_COUNT": "firstrun_nfix",
        "WORD_FIXATION_COUNT": "nfix"
    }

    gaze_features += ["WORD_FIRST_RUN_DURATION"]

    print(df[gaze_features].corr())

    df.dropna()

    reader_grouped_df = df.groupby(["PP_NR"])[gaze_features].mean().reset_index(level=0)

    print(df["PP_NR"].unique())

    medoids = clusterize_user_profiling(reader_grouped_df, gaze_features)

    print(medoids)

    datasets = []

    for user in medoids:
        datasets.append(df[df.PP_NR == user].reset_index(drop=True))


    # use last 16 Trial as test
    import random
    TRIAL_TESTS = random.sample(list(range(5, 165)), 20)

    print("selected random trials: ", TRIAL_TESTS)

    for user, df in zip(medoids, datasets):

        df["WORD_ID"] = df.index
        df["WORD_ID"] = df["WORD_ID"].astype(float)
        df["PART"] = df["PART"].astype(float)
        df["TRIAL"] = df["TRIAL"].astype(float)
        df.rename(columns=mapping_columns, inplace=True)

        test_idx = df.sentnum.isin(TRIAL_TESTS)
        dev_idx = ~test_idx

        test_df = df[test_idx]
        dev_df = df[dev_idx]

        dev_df.to_csv(f"augmenting_nlms_meco_data/geco/{user}_dataset.csv")
        test_df.to_csv(f"augmenting_nlms_meco_data/geco/{user}_dataset_test.csv")


if __name__ == "__main__":
    main()