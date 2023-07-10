from segment import *
from graph import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import logging

import os

# 현재 날짜와 시간을 가져옴
now = datetime.now()

# 한국 시간대로 변환
korea_timezone = pytz.timezone("Asia/Seoul")
korea_time = now.astimezone(korea_timezone)

# 날짜 문자열 추출
korea_date = korea_time.strftime("%Y-%m-%d")


# 로그 파일 이름에 현재 시간을 포함시킵니다.
log_filename = f'./logs/model_result/elastic_program_{korea_date}.log'


# 로깅 핸들러를 생성합니다.
log_handler = logging.FileHandler(log_filename)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

# 로거를 생성하고 로깅 핸들러를 추가합니다.
logger = logging.getLogger(f's')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def kmeans_modeling():
    # 현재 폴더 경로 확인
    current_path = os.getcwd() + "/"
    
    folder_path = "train_models"
    os.makedirs(folder_path, exist_ok=True)
    
    data = get_index_data('describe*').reset_index(drop=True)

    select_columns = ["평균 접속 수(1분)", "차단율(%)", "평균 접속 간격(초)", "접속 UA 수", "최다 이용 UA 접속 비율(%)", "최대 빈도 URL 접속 비율(%)"]

    select_data = data[select_columns]
    
    # 표준화
    scaler = StandardScaler()
    select_data_scaled = pd.DataFrame(scaler.fit_transform(select_data), columns=select_columns)

    pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정

    printcipalComponents = pca.fit_transform(select_data_scaled)

    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2'])
    # # 주성분으로 이루어진 데이터 프레임 구성

    n_clusters=5
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, random_state=0)

    kmeans_model = kmeans.fit(principalDf[['component1', 'component2']].values)
    
    # kmeans 모델 저장 경로
    model_path =  current_path + f'{folder_path}/kmeans_model.pkl'

    # 모델 저장
    with open(model_path, 'wb') as file:
        pickle.dump(kmeans_model, file)

    # PCA 모델 저장 경로
    pca_model_path = current_path + f'{folder_path}/pca_model.pkl'

    # PCA 모델 저장
    with open(pca_model_path, 'wb') as file:
        pickle.dump(pca, file)


    # 스케일러 저장 경로
    scaler_path = current_path  + f'{folder_path}/scaler.pkl'

    # 스케일러 저장
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)

def import_model():
       
    # 현재 폴더 경로 확인
    current_path = os.getcwd() + "/"
    
    folder_path = "train_models"
    os.makedirs(folder_path, exist_ok=True)
    
    # 모델 파일 경로
    model_path = current_path + f'{folder_path}/kmeans_model.pkl'

    # 모델 불러오기
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # PCA 모델 파일 경로
    pca_model_path = current_path + f'{folder_path}/pca_model.pkl'

    # PCA 모델 불러오기
    with open(pca_model_path, 'rb') as file:
        loaded_pca = pickle.load(file)
        
    # 스케일러 파일 경로
    scaler_path = current_path + f'{folder_path}/scaler.pkl'

    # 스케일러 불러오기
    with open(scaler_path, 'rb') as file:
        loaded_scaler = pickle.load(file)
    return loaded_model, loaded_pca, loaded_scaler
        
def return_labels(data):
    loaded_model, loaded_pca, loaded_scaler = import_model()
    select_columns = ["평균 접속 수(1분)", "차단율(%)", "평균 접속 간격(초)", "접속 UA 수", "최다 이용 UA 접속 비율(%)", "최대 빈도 URL 접속 비율(%)"]

    select_data = data[select_columns]
    
    select_data_scaled = pd.DataFrame(loaded_scaler.transform(select_data), columns=select_columns)
    
    printcipalComponents = loaded_pca.transform(select_data_scaled)

    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2'])
    
    # # 주성분으로 이루어진 데이터 프레임 구성
    cluster_label = loaded_model.predict(principalDf[['component1', 'component2']])
    
    
    return list(cluster_label)




def return_devid_cluster_label(dec_data):
    cluster_label = return_labels(dec_data)
    return cluster_label    
    

    
def return_total_label_to_elasticsearch(dev_id):
    dec_data = get_final_dec_data_dev_id(dev_id)
    
    # 차단 조건 
    if dec_data["평균 접속 수(1분)"].values >= 100 and dec_data["차단 수"].values >= 50 and dec_data["최다 이용 UA 접속 비율(%)"].values >= 90 and dec_data["최대 빈도 URL 접속 비율(%)"].values >= 90:
        logger.info(f"{dev_id} : Rule 1 matched!")
        save_db_data(dec_data, "abnormal_describe")
        
    elif dec_data["최다 접속 URL"].values == "123.57.193.95" or dec_data["최다 접속 URL"].values == "123.57.193.52":
        logger.info(f"{dev_id} : Rule 2 matched!")
        save_db_data(dec_data, "abnormal_describe")
    
    else: 
        label = return_devid_cluster_label(dec_data)[0]
        if label == 2:
            logger.info(f"{dev_id}(label = 2) : Rule 3 matched!")
            save_db_data(dec_data, "abnormal_describe")
            return 
        logger.info(f"{dev_id}(label = {label}) : Normal User")


def process():
    # 랜덤 샘플링 버전
    # dev_id_list = get_sDevID_random("1m")

    # 허용 block 혼합 버전
    dev_id_list = get_pass_block_dev_list()

    for dev_id in dev_id_list:
        return_total_label_to_elasticsearch(dev_id)


# 실루엣 계수
def visualize_sil(List, X):
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    for n_clusters in List:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters + \
                          '\nSilhouette Score :' + str(round(silhouette_avg,3)) ,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()
    