from segment import *
from graph import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle
import logging
import plotly.express as px
import yaml

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# 현재 날짜와 시간을 가져옴
now = datetime.now()

current_path = os.getcwd() + "/"

# 한국 시간대로 변환
korea_timezone = pytz.timezone("Asia/Seoul")
korea_time = now.astimezone(korea_timezone)

# 날짜 문자열 추출
korea_date = korea_time.strftime("%Y-%m-%d")


# 로그 파일 이름에 현재 시간을 포함시킵니다.
try:
    log_filename = current_path + f'logs/model_result/elastic_program_{korea_date}.log'

except:
    log_filename = f'code/logs/model_result/elastic_program_{korea_date}.log'
    
# 로깅 핸들러를 생성합니다.
log_handler = logging.FileHandler(log_filename)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

# 로거를 생성하고 로깅 핸들러를 추가합니다.
logger = logging.getLogger(f's')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


# 변수 선택 
select_columns = ['평균 접속 수(1분)', '최다 이용 UA 접속 수', '최대 빈도 URL 접속 횟수', '평균 접속 횟수(1개 URL)', '최대 연속 URL 접속 횟수', '고유 접속 URL 수', '평균 패킷 길이']

def train_data_preprocessing(data):
    # 중복되는 가입자 ID 삭제, 가장 최근 기록만 남김
    data = data.drop_duplicates(subset="가입자 ID", keep='last')
    data.reset_index(drop=True, inplace=True)
    
    # 접속 시간이 0인 데이터 삭제 
    zero_connect_time_index = data[data['접속 시간(분)'] == 0].index
    data = data.drop(zero_connect_time_index)

    # 평균 접속 시간이 0인 데이터 삭제 
    zero_connect_count_index = data[data['평균 접속 수(1분)'] == 0.0].index
    data = data.drop(zero_connect_count_index)
    
    data.reset_index(drop=True, inplace=True)
    

    
    return data

# 데이터 표준화
def standard_transfrom(data):
    X  = data.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
    
    

def pca_auto_choice(data): 
    try:
        X = data.values
    except:
        X = data
    pca = PCA()

    pca.fit(X)
    
    # 주성분의 분산 설명 비율 출력
    explained_variance_ratio = pca.explained_variance_ratio_
    sum_variance = 0 
    for i, ratio in enumerate(explained_variance_ratio):
        sum_variance += ratio
        if sum_variance >= 0.90:
            print(f'best pca count : {i+1}')
            result_pca_count = i+1
            break 
    variables = data.columns
    pca_columns = []
    for pca in range(result_pca_count):
        pca_columns.append(f'component {pca+1}')
    pca = PCA(n_components=result_pca_count)
    
    printcipalComponents = pca.fit_transform(X)
    
    # 주성분(PC)과 원본 변수 간의 관련성 출력
    components = pca.components_
    for i, pc in enumerate(components):
        print(f"PC{i+1}과 원본 변수 간의 관련성:")
        for j, var in enumerate(variables):
            print(f"{var}: {pc[j]}")
        print()

    principalDf = pd.DataFrame(data=printcipalComponents, columns = pca_columns)

    return principalDf

def pca_num_choice(data, num_components): 
    variables = data.columns
    # 현재 폴더 경로 확인
    
    folder_path = "train_models"
    os.makedirs(folder_path, exist_ok=True)
    
    pca_columns = []
    for pca in range(num_components):
        pca_columns.append(f'component{pca+1}')
        
    try:
        X = data.values
    except:
        X = data
    pca = PCA(n_components=num_components)

    printcipalComponents = pca.fit_transform(X)
    
    # 주성분(PC)과 원본 변수 간의 관련성 출력
    components = pca.components_
    for i, pc in enumerate(components):
        print(f"PC{i+1}과 원본 변수 간의 관련성:")
        for j, var in enumerate(variables):
            print(f"{var}: {pc[j]}")
        print()

        
    explained_variance_ratio = pca.explained_variance_ratio_
    
    for i, ratio in enumerate(explained_variance_ratio):
        print(f'component {i+1} Ratio : {ratio}')
       
    
    principalDf = pd.DataFrame(data=printcipalComponents, columns = pca_columns)
    
    
    # PCA 모델 저장 경로
    pca_model_path = current_path + f'{folder_path}/pca_model.pkl'

    # PCA 모델 저장
    with open(pca_model_path, 'wb') as file:
        pickle.dump(pca, file)
        
    print(pca_model_path, 'PCA 저장 완료')

    
    return principalDf 

def standard_transfrom(data):
    X  = data.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
    




def kmeans_modeling(k):
    
    folder_path = "train_models"
    os.makedirs(folder_path, exist_ok=True)
    
    # 학습용 데이터 불러오기
    data = get_index_data('describe*').reset_index(drop=True)
    
    # 학습용 데이터 전처리(중복 데이터 삭제, 접속시간=0, 접속수=0 데이터 삭제)
    data = train_data_preprocessing(data)
    
    

    data = data[select_columns]
    
    # pca 진행
    principalDf = pca_num_choice(data, 2)
    
    # 모델 적합
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=0)
    kmeans_model = kmeans.fit(principalDf[['component1', 'component2']].values)
    
    print(f'{len(data)} Data success train!')
    print()
    
    # 이상치 군집 저장
    principalDf['kmeans_label'] = kmeans_model.fit_predict(principalDf[['component1', 'component2']].values)
    
    
    outlier_k = principalDf['kmeans_label'].value_counts().index[-1]
    print(f'outlier k = {outlier_k}')
    
    print('k 별 count')
    print(principalDf['kmeans_label'].value_counts())
    print()
    
    
    # YAML 파일 경로
    outlier_k_path = current_path + "train_models/kmeans_outlier_k.yaml"

    
    
    # YAML 데이터 생성
    yaml_data = {
        "kmeans_outlier_k": str(outlier_k), 
        "k" : str(k)
    }

    # YAML 파일 작성
    with open(outlier_k_path, "w") as f:
        yaml.safe_dump(yaml_data, f)
    
    print(outlier_k_path, '저장 완료')
    
    
    # kmeans 모델 저장 경로
    model_path =  current_path + f'{folder_path}/kmeans_model.pkl'
    
    

    # 모델 저장
    with open(model_path, 'wb') as file:
        pickle.dump(kmeans_model, file)
    
    print(model_path, '모델 저장 완료')
    # Scatter plot 그리기
    fig = px.scatter(principalDf, x='component1', y='component2', color=principalDf['kmeans_label'])



    # HTML 파일로 저장
    fig.write_html("kmeans_scatter_plot.html")
    print("/kmeans_scatter_plot.html", 'cluster plot 저장 완료')
#     # 스케일러 저장 경로
#     scaler_path = current_path  + f'{folder_path}/scaler.pkl'

#     # 스케일러 저장
#     with open(scaler_path, 'wb') as file:
#         pickle.dump(scaler, file)

def import_model():
       
    # 현재 폴더 경로 확인
    current_path = os.getcwd() + "/"
    
    folder_path = "train_models"
    os.makedirs(folder_path, exist_ok=True)
    
    
    # kmeans 불러오기
    # 모델 파일 경로
    kmeans_model_path = current_path + f'{folder_path}/kmeans_model.pkl'

    # 모델 불러오기
    with open(kmeans_model_path, 'rb') as file:
        kmeans_loaded_model = pickle.load(file)

    # PCA 모델 파일 경로
    pca_model_path = current_path + f'{folder_path}/pca_model.pkl'

    # PCA 모델 불러오기
    with open(pca_model_path, 'rb') as file:
        loaded_pca = pickle.load(file)
        
#     # 스케일러 파일 경로
#     scaler_path = current_path + f'{folder_path}/scaler.pkl'

#     # 스케일러 불러오기
#     with open(scaler_path, 'rb') as file:
#         loaded_scaler = pickle.load(file)
    return  kmeans_loaded_model, loaded_pca
 
    
def kmeans_predict(data, model):
    # # 주성분으로 이루어진 데이터 프레임 구성
    cluster_label = model.predict(data.values)
    
    return cluster_label
    
def return_labels(data):
    kmeans_loaded_model, loaded_pca  = import_model()
    

    select_data = data[select_columns]
    
    
    printcipalComponents = loaded_pca.transform(select_data)

    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2'])
    
    
    # kmeans
    kmeans_label = kmeans_predict(principalDf, kmeans_loaded_model)
    
    
 
    #dbscan_label = 
    
    return kmeans_label




# def return_devid_cluster_label(dec_data):
#     cluster_label = return_labels(dec_data)
#     return cluster_label    
    
def get_kmeans_outlier_k():
    # YAML 파일 경로
    outlier_k_path_kmeans = current_path + "train_models/kmeans_outlier_k.yaml"
    
    # YAML 파일 읽기
    with open(outlier_k_path_kmeans, "r") as f:
        yaml_data = yaml.safe_load(f)


    kmeans_outlier_k = int(yaml_data.get("kmeans_outlier_k"))
    
    
    
    return kmeans_outlier_k


def rule_based_modeling(dec_data, dev_id):
    if dec_data["평균 접속 수(1분)"].values >= 100 and dec_data["차단 수"].values >= 50 and dec_data["최다 이용 UA 접속 비율(%)"].values >= 90 and dec_data["최대 빈도 URL 접속 비율(%)"].values >= 90:
        logger.info(f"{dev_id} : Rule 1 matched!")
        save_db_data(dec_data, "abnormal_describe")
        return 
    elif dec_data["최다 접속 URL"].values == "123.57.193.95" or dec_data["최다 접속 URL"].values == "123.57.193.52":
        logger.info(f"{dev_id} : Rule 2 matched!")
        save_db_data(dec_data, "abnormal_describe")
        return


def return_total_label_to_elasticsearch(dev_id):
    
    # kmeams outlier 
    kmeans_outlier_k = get_kmeans_outlier_k()
    
    
    dbscan_outlier_k = -1
    
    dec_data = get_final_dec_data_dev_id(dev_id)    
    
    # rule based model 
    rule_based_modeling(dec_data, dev_id)
    
    
    # kmeans 
    kmeans_label = return_labels(dec_data)[0]
    
    
    if kmeans_label == kmeans_outlier_k:
        logger.info(f"{dev_id}(label = {kmeans_outlier_k}) : Rule 3 matched!(kmeans)")
        save_db_data(dec_data, "abnormal_describe")
        return 
    # dbscan
    

    logger.info(f"{dev_id}(label = {kmeans_label}) : Normal User")
               

def process():
    # 랜덤 샘플링 버전
    # dev_id_list = get_sDevID_random("1m")

    # 허용 block 혼합 버전
    dev_id_list = get_pass_block_dev_list()

    for dev_id in dev_id_list:
        return_total_label_to_elasticsearch(dev_id)

# 데이터를 로드하고 전처리한 후 X에 저장
# 최대 클러스터 개수 설정
def clustering_choice_k_scree(data, max_clusters):
    X = data.values 
    max_clusters = max_clusters 

    # 클러스터 개수에 따른 응집력 또는 분산 설명 비율 계산
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # 스크리 플롯 그리기
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia or Variance Explained')
    plt.title('Scree Plot')
    plt.show()


def visualize_sil(data, max_clusters):
    X = data.values
    List = [i for i in range(2, max_clusters+1)]
    
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
    