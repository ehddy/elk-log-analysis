# 기본
import os 
import pytz
from datetime import datetime, timedelta
import pandas as pd
import yaml
import requests
import json
import numpy as np
import random
import pickle

# Elasticsearch
from elasticsearch import Elasticsearch
from elastic_transport import ConnectionTimeout


# 시각화
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

# 머신러닝 모델
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 경고창 제거, 로깅
import warnings
import logging
warnings.filterwarnings('ignore')


class Elk: 
    # 현재 날짜와 시간을 가져옴

    def __init__(self):
        now = datetime.now()
        # 한국 시간대로 변환
        korea_timezone = pytz.timezone("Asia/Seoul")
        self.korea_time = now.astimezone(korea_timezone)

        # 날짜 문자열 추출
        self.korea_date = self.korea_time.strftime("%Y-%m-%d")


        self.current_path = os.getcwd() + "/"

        # 로그 파일 이름에 현재 시간을 포함시킵니다.

        self.log_folder = self.current_path + 'logs/save_elasticsearch/'
        os.makedirs(self.log_folder, exist_ok=True)

        self.log_filename = self.log_folder + f'elastic_program_{self.korea_date}.log'

        # 로깅 핸들러를 생성합니다.
        self.log_handler = logging.FileHandler(self.log_filename)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

        # 로거를 생성하고 로깅 핸들러를 추가합니다.
        self.logger = logging.getLogger('s')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)



        with open(self.current_path + 'config.yaml', encoding='UTF-8') as f:
            _cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.elasticsearch_ip = _cfg['ELASTICSEARCH_IP_ADDRESS']  

               
        # Elasticsearch 연결
        self.es = Elasticsearch([self.elasticsearch_ip])

    # UTC 시간을 한국 시간으로 변환하는 함수
    def utc_to_kst(self, utc_time):
        kst = pytz.timezone('Asia/Seoul')
        kst_time = utc_time.astimezone(kst)
        return kst_time




    # 1. get(데이터 수집 관련 함수)

    # 1) 가입자 리스트 추출 

    # 랜덤하게 가입자 추출
    def get_sDevID_random(self, time_choice):
 

        # Elasticsearch에서 데이터 검색
        res = self.es.search(
            index='sa*',
            body={
                "size": 0,
                "aggs": {
                    "devid_count": {
                        "terms": {
                            "field": "sDevID.keyword",
                            "size": 10000,
                            "exclude": "?"
                        }
                    }
                },
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": f"now-{time_choice}",
                            "lt": "now"
                        }
                    }
                }
            }
        )

        # 결과에서 'devid_count' 어그리게이션 버킷 추출
        sDevID_buckets = res['aggregations']['devid_count']['buckets']

        # 'sDevID'를 저장할 리스트 생성
        sDevID_list = []

        # 각 버킷에서 'sDevID' 값을 추출하여 리스트에 저장
        for bucket in sDevID_buckets:
            sDevID_list.append(bucket['key'])

        # 랜덤으로 100개의 값을 선택하여 추출
        random_sDevID_list = random.sample(sDevID_list, 100)


        return random_sDevID_list

    # 차단 허용 기준으로 가입자 추출
    def get_sDevID(self, cRes, min_count, time_choice):

        # Elasticsearch에서 데이터 검색
        res = self.es.search(
            index='sa*',
            body={
            "size": 0,
            "aggs": {
                "devid_count": {
                    "terms": {
                        "field": "sDevID.keyword",
                        "min_doc_count": min_count,
                        "size": 10000,
                        "exclude": "?"
                    }
                }
            },
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": f"now-{time_choice}",
                                    "lt": "now"
                                }
                            }
                        },
                        {
                            "term": {
                                "cRes.keyword": cRes
                            }
                        }
                    ]
                }
            }
        })

        # 결과에서 'devid_count' 어그리게이션 버킷 추출
        sDevID_buckets = res['aggregations']['devid_count']['buckets']

        # 'sDevID'를 저장할 리스트 생성
        sDevID_list = []

        # 각 버킷에서 'sDevID' 값을 추출하여 리스트에 저장
        for bucket in sDevID_buckets:
            sDevID_list.append(bucket['key'])
        
        return sDevID_list

    def get_pass_block_dev_list(self):
        block_list = self.get_sDevID("차단", 1, '30m')
        self.get_sDevID_random("1m")
        pass_list = self.get_sDevID('허용', 500, '30m')
        pass_list = random.sample(pass_list, 100)
        # random_list = self.get_sDevID_random('30m')
        
        total_dev_list = list(set(block_list + pass_list))
        
        return total_dev_list

    # 키워드 기준으로 가입자 추출(100명)
    def get_keywords_match_devid(self, column_name, match_keywords, time_choice, random_option=False): 
        wildcard_queries = []
        for keyword in match_keywords:
            wildcard_query = {
                "wildcard": {
                    column_name: {
                        "value": f"*{keyword}*"
                    }
                }
            }
            wildcard_queries.append(wildcard_query)
        
        # Elasticsearch에서 집계된 데이터 검색
        res = self.es.search(
            index='sa*',
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "bool": {
                                    "should": wildcard_queries,
                                    "must_not": {
                                        "term": {
                                            "sDevID.keyword": "?"  # 특정 값으로 설정하세요
                                        }
                                    }
                                }
                            },
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": f"now-{time_choice}",
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 0,  # 집계된 결과만 가져옴
                "aggs": {
                    "top_devid": {
                        "terms": {
                            "field": "sDevID.keyword",
                            "size": 10000,
                            "order": {"_count": "desc"}
                        }
                    }
                }
            }
        )
        
        buckets = res["aggregations"]["top_devid"]["buckets"]
        devid_list = [bucket["key"] for bucket in buckets]
        
        if random_option == True:

            # 랜덤 추출
            devid_list = random.sample(devid_list, 100)
        else:
            # 빈도순
            devid_list = devid_list[:100]
        return devid_list

    # Elasticsearch에서 데이터 추철

    # 인덱스 이름을 기준으로 모든 데이터 추출 
    def get_index_data(self, index_name):

        # 가져올 인덱스 이름 설정
        index_name = '{}*'.format(index_name)

        # Elasticsearch에서 모든 데이터 검색
        res = self.es.search(index=index_name, body={"query": {"match_all": {}}}, size=10000, scroll='10m')


        data = pd.DataFrame([hit['_source'] for hit in res['hits']['hits']])
        
        
        scroll_id = res['_scroll_id']
        hits_list = []
        
        while True:
            res = self.es.scroll(scroll_id=scroll_id, scroll='10m')
            if len(res['hits']['hits']) == 0:
                break
            hits = res['hits']['hits']
            hits_list.extend(hits)
        
        self.es.clear_scroll(scroll_id=scroll_id)
        data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
        data = pd.concat([data, data_scroll])

        data.reset_index(drop=True, inplace=True)
        
        return data
        



    # 가입자 이름을 기준으로 모든 데이터 추출 
    def get_sDevID_data(self, user_id):
        # Elasticsearch에서 데이터 검색
        res = self.es.search(
            index='sa*',
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "sDevID.keyword": user_id  # 해당 유저의 아이디로 필터링
                                }
                            },
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": "now-1h",
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 10000
            },
            scroll='10m'  # 스크롤 유지 시간 설정
        )
        
        # 첫 번째 스크롤 결과 가져오기
        data = pd.DataFrame([hit['_source'] for hit in res['hits']['hits']])
        
        
        scroll_id = res['_scroll_id']
        hits_list = []
        
        while True:
            res = self.es.scroll(scroll_id=scroll_id, scroll='10m')
            if len(res['hits']['hits']) == 0:
                break
            hits = res['hits']['hits']
            hits_list.extend(hits)
        
        self.es.clear_scroll(scroll_id=scroll_id)
        data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
        data = pd.concat([data, data_scroll])

        
        return data


    # 특정 키워드 1개를 기준으로 포함 데이터 가져오기
    def get_category_match_data(self, column_name, match_keyword):
        res = self.es.search(
            index='sa*',
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "match": {
                                    column_name : match_keyword  # 
                                }
                            },
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": "now-1h",
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 10000
            },
            scroll='10m'  # 스크롤 유지 시간 설정
        )
        
        # 첫 번째 스크롤 결과 가져오기
        data = pd.DataFrame([hit['_source'] for hit in res['hits']['hits']])
        
        scroll_id = res['_scroll_id']
        hits_list = []
        
        while True:
            res = self.es.scroll(scroll_id=scroll_id, scroll='10m')
            if len(res['hits']['hits']) == 0:
                break
            hits = res['hits']['hits']
            hits_list.extend(hits)
        
        self.es.clear_scroll(scroll_id=scroll_id)
        data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
        data = pd.concat([data, data_scroll])

        return data

    # 모델 통합용(전처리 통합)
    def get_final_dec_data_dev_id(self,dev_id):
        dev_data = self.get_sDevID_data(dev_id)
        scaled_data = self.preprocessing_data(dev_data)
        dec_data = self.describe(scaled_data)
        
        return dec_data
    
    def get_abnormal_data_show(self):
        data = self.get_index_data('abnormal_describe*')
        id_count_dic = data.groupby('가입자 ID')['전체 접속 횟수'].count().sort_values(ascending=False).to_dict()        
        data['비정상 판별 횟수'] = data['가입자 ID'].apply(lambda x : id_count_dic[x])
        data = data.sort_values(by='timestamp', ascending=False)  
        data = data.drop_duplicates(subset='가입자 ID', keep='first') 
        data = data.sort_values(by=['비정상 판별 횟수', '판별 등급', '평균 접속 수(1분)'], ascending=[False, True, False])
        data.reset_index(drop=True, inplace=True)
        data.drop('timestamp', axis=1, inplace=True)
        data = data[['가입자 ID', '비정상 판별 횟수','판별 등급', '판별 요인', '평균 접속 수(1분)', '차단율(%)','최다 접속 URL', '최다 접속 IP 위치', '접속 기간']]
        data.rename({'접속 기간': '의심 접속 시간'}, axis=1, inplace=True)

        return data

    # 3) 데이터 통합용 

    # 허용, 차단을 기준으로 최종 데이터 추출 (szie는 추출하고 싶은 가입자의 수)
    def get_data(self, size, cRes):
        sDevID_list = self.get_sDevID(size, cRes)  
        total_df_list = []
        for dev_id in sDevID_list:
            data = self.get_sDevID_data(dev_id)
            scaled_data = self.preprocessing_data(data)
            total_df_list.append(scaled_data)
            self.save_csv(new_data, dev_id)
        
        return total_df_list

    def delete_data_by_id(self, index_pattern, user_id):

        # 삭제할 데이터를 찾기 위한 쿼리
        query = {
            "query": {
                "term": {
                    "가입자 ID.keyword": user_id
                }
            } , "size": 10000
        }
        try:
            # 인덱스 패턴과 일치하는 모든 인덱스를 가져옵니다.
            matching_indices = self.es.indices.get_alias(index=index_pattern).keys()

            total_deleted = 0
            for index_name in matching_indices:
                # 쿼리를 실행하여 가입자 ID에 해당하는 데이터를 찾습니다.
                result = self.es.search(index=index_name, body=query)

                # 검색 결과에서 삭제 대상 문서의 ID를 가져옵니다.
                doc_ids = [hit['_id'] for hit in result['hits']['hits']]

                # 삭제 대상 문서들을 삭제합니다.
                for doc_id in doc_ids:
                    self.es.delete(index=index_name, id=doc_id)

                total_deleted += len(doc_ids)

            print(f"총 {total_deleted}개의 문서가 삭제되었습니다.")
        except Exception as e:
            print(f"데이터 삭제 중 오류가 발생했습니다: {e}")

    # 4) 집계용 

    # 1시간 동안의 유저 데이터를 요약 집계 
    def describe(self, data):
        
        # 추출한 접속 일자
        connect_date = data.index[0].strftime("%Y-%m-%d")
        
        # 추출 시간
        start = data.index[0]
        end = data.index[-1]
        
        # 접속 기간
        connect_date_hour_start = start.strftime("%H시 %M분")
        connect_date_hour_end = end.strftime("%H시 %M분")
        connect_duration = connect_date_hour_start + " ~ " + connect_date_hour_end
        

        # 접속 시간(분)
        connect_time = (end - start)
        connect_minutes = int(connect_time.total_seconds() // 60)
        
        # 지속시간
        duration_mean = round(data['duration'].mean(), 2)
        # duration_mean = str(duration_mean) + "초"

        # duration_min = round(data['duration'].min(), 2)
        # duration_min = str(duration_min) + "초"

        duration_max = round(data['duration'].max(), 2)
        # duration_max_minute = int(duration_max // 60)
        # duration_max_second = int(duration_max % 60)
        # duration_max_time = "{:}분 {}초".format(duration_max_minute, duration_max_second)

        
    
        # count 
        connect_count = len(data) 
        
        # 고유 접속 URL수 
        connect_url_count = len(data['sHost'].unique())
        
        # 최다 접속 URL
        main_url = data['sHost'].value_counts().index[0]
        
        # 최다 접속 IP 주소
        main_ip = data['uDstIp'].value_counts().index[0]
        
        main_ip_country = self.get_geolocation(main_ip)[0]
        
            
        # 평균 접속 수(1분)
        if connect_minutes == 0:
            mean_connect_url_count = connect_count 
            
        else:
            mean_connect_url_count = int(connect_count / connect_minutes)
        
        
        
        
        # 접속 기기 수 
        connect_ua_count = len(data['sUA'].unique())
        
        
        # 최다 이용 기기 
        main_ua = data['sUA'].value_counts().index[0]
        
        # 최다 이용 기기 접속 수 
        main_ua_connect_count = data['sUA'].value_counts()[0] 
        

        # 차단 횟수
        block_count = data[data['cRes'] == '차단'].shape[0]
        
        # 차단률
        block_ratio = round((block_count / connect_count) * 100, 2)
        
        # 동일 URL 연속 접속 횟수
        succession_url_connect_count = data['sHost_same_as_previous_count'].max() + 1
        
        # 평균 패킷 길이
        mean_packet_lenth = round(data.nSize.mean(), 2)
        
            
        # 최다 접속 포트 번호
        main_port = data['uDstPort'].value_counts().index[0]

        
        # 최대 빈도 URL 접속 횟수 
        max_frequent_url_connect_count = data['sHost'].value_counts()[0]
        decribe_list = [data['sDevID'][0],connect_count,  mean_connect_url_count, block_count,  block_ratio, duration_mean, duration_max, 
                        connect_url_count, succession_url_connect_count, main_url, main_ip, main_ip_country,
                        max_frequent_url_connect_count,connect_ua_count, main_ua, main_ua_connect_count
                        ,connect_date, connect_duration, connect_minutes, 
                    mean_packet_lenth,  main_port]
        
        
        column_lists = ['가입자 ID','전체 접속 횟수', '평균 접속 수(1분)','차단 수', '차단율(%)','평균 접속 간격(초)', '최장 접속 간격(초)', '고유 접속 URL 수', '최대 연속 URL 접속 횟수', 
                    '최다 접속 URL', '최다 접속 IP','최다 접속 IP 위치', '최대 빈도 URL 접속 횟수' ,'접속 UA 수', '최다 이용 UA', 
                        '최다 이용 UA 접속 수' ,'접속 일자', '접속 기간', '접속 시간(분)', '평균 패킷 길이', 
                    '최다 접속 포트 번호']
        describe_data  = pd.DataFrame([decribe_list], columns=column_lists)
        
        
        # 비율이 높을수록 사용자가 다양한 URL을 방문했음을 의미
        describe_data['접속 횟수 대비 고유 URL 비율(%)'] = round((describe_data['고유 접속 URL 수'] / describe_data['전체 접속 횟수']) * 100, 2)

        # 이 값이 높을수록 사용자가 특정 URL에 집중적으로 접속
        describe_data['평균 접속 횟수(1개 URL)'] = round(describe_data['전체 접속 횟수'] / describe_data['고유 접속 URL 수'], 2)


        # 이 값이 높을수록 사용자는 특정 이용 기기의 사용률이 높음
        describe_data['최다 이용 UA 접속 비율(%)'] = round((describe_data['최다 이용 UA 접속 수'] / describe_data['전체 접속 횟수']) * 100, 2)


        # 최다 연속 URL 접속 비율
        # 이 값이 높을수록 사용자는 특정 이용 기기의 사용률이 높음
        describe_data['최다 연속 URL 접속 비율(%)'] = round((describe_data['최대 연속 URL 접속 횟수'] / describe_data['전체 접속 횟수']) * 100, 2)


        # 최대 빈도 URL 접속 비율
        # 이 값이 높을수록 사용자는 특정 이용 기기의 사용률이 높음
        describe_data['최대 빈도 URL 접속 비율(%)'] = round((describe_data['최대 빈도 URL 접속 횟수'] / describe_data['전체 접속 횟수']) * 100, 2)



        return describe_data

    # 카테고리별로 집계 
    def describe_category(self, data, column_name):
        unique_list = list(data[column_name].unique())
        categoty_segment_data = []
        dev_id = data['sDevID'][0]
        unique_describe = []
        for name in unique_list:
            category_data = data.loc[data[column_name] == name, :]
            categoty_segment_data.append(category_data)
            dec = self.describe(new_category_data)
            dec[column_name] = name
            unique_describe.append(dec)
            

        # 데이터프레임들을 수직으로 합치기
        merged_unique_data = pd.concat(unique_describe, axis=0)
        merged_unique_data['가입자 ID'] = dev_id
        # merged_unique_data.set_index([column_name, inplace=True)
        merged_unique_data.drop(columns=['접속 일자', '접속 기간', '최다 이용 UA', '접속 UA 수', '최다 이용 UA 접속 수', '최다 이용 UA 접속 비율(%)'], axis=1, inplace=True)
        
        merged_unique_data = merged_unique_data.sort_values('전체 접속 횟수', ascending=False)
        merged_unique_data.reset_index(drop=True, inplace=True)
        
        return merged_unique_data, categoty_segment_data

    
    def seasonal_decompose_values(self, data, period=5):

        data['date_group_minute'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        # 데이터를 시간적으로 그룹화하여 그룹별 통계 계산
        grouped_data = data.groupby('date_group_minute')[['sHost']].count()

        result = sm.tsa.seasonal_decompose(grouped_data['sHost'], model='additive', period=period)

        original_values = grouped_data['sHost'].values
        trend_values = result.trend.values
        seasonal_values = result.seasonal.values
        residual_values = result.resid.values
        
        
        return seasonal_values


    # 5) 기타 데이터 추가용도

    # 어떤 키워드가 가장 높은 비율로 나타났는지
    def get_top_keyword_frequency(self, data, column_name, keywords):
        keyword_counts = {}
        
        for keyword in keywords:
            keyword_counts[keyword] = data[column_name].str.contains(keyword, case=False).sum()
            
        keyword_df = pd.DataFrame({'Keyword': list(keyword_counts.keys()), 'Frequency': list(keyword_counts.values())})
        keyword_df['Percentage'] = (keyword_df['Frequency'] / len(data)) * 100
        keyword_df = keyword_df.sort_values('Frequency', ascending=False)
        
        top_keyword = keyword_df.iloc[0]['Keyword']
        top_frequency = keyword_df.iloc[0]['Frequency']
        top_percentage =  keyword_df.iloc[0]['Percentage'].round(2)
        
        return top_keyword, top_frequency, top_percentage        
            
            
    # ip 주소의 지리적 위치 
    def get_geolocation(self, ip_address):
        url = f"http://ip-api.com/json/{ip_address}"
        response = requests.get(url)
        data = json.loads(response.text)
        
        if data["status"] == "fail":
            return ['None', 'None', 'None']
        
        country = data["country"]
        region = data["regionName"]
        city = data["city"]
        
        
        return [country, region, city]

    # ip 주소 지리적 위치 집계 
    def get_ip_describe(self, data):
        host_list = data.sHost.value_counts()[:10].index
        describe_url_df = data.groupby(['sHost', 'uDstIp', 'sUA'])[['cRes']].count().reset_index().rename({'cRes' : 'connect_count'}, axis=1)
        describe_url_df = describe_url_df[describe_url_df['sHost'].isin(host_list)].sort_values('connect_count', ascending=False).reset_index(drop=True)
        ip_list  = describe_url_df['uDstIp'].values
        ip_location_list = []
        for ip in ip_list:
            location = self.get_geolocation(ip)
            ip_location_list.append(location)

        describe_url_df[['country', 'regionName', 'city']] = ip_location_list
        
        return describe_url_df
    
    

    # 랜덤 가입자 집계 데이터 저장 
    def save_db_random_devid(self):
        block_list = self.get_sDevID("차단", 1, '30m')
        sDevID_list = self.get_sDevID_random("1m")
        
        total_dev_list = list(set(block_list + sDevID_list))
        for dev_id in total_dev_list:
            try:
                dec_data = self.get_final_dec_data_dev_id(dev_id)
                
                if dec_data['접속 시간(분)'].values <= 30 or dec_data['평균 접속 수(1분)'].values <= 1:
                    self.logger.info(f"not save {dev_id} data(index name : {index_name}, information short)")
                    continue
                
                # 데이터프레임을 Elasticsearch 문서 형식으로 변환
                index_name =  'describe'
                self.save_db_data(dec_data, index_name)
                self.logger.info(f"save success {dev_id} data(index name : {index_name})")
            except Exception as e:
                self.logger.error(e)
                continue
            
        
    # 집계 데이터 특정 index_name에 저장 
    def save_db_data(self, dec_data, index_name):

        # 데이터프레임을 Elasticsearch 문서 형식으로 변환
        documents = dec_data.to_dict(orient='records')

        # Elasticsearch에 데이터 색인
        index_name = f'{index_name}-{self.korea_date}'  # 저장할 인덱스 이름
        for doc in documents:
            self.es.index(index=index_name, body=doc)
    
    
    def preprocessing_keywords_match_data(self,  dev, keyword_lists):
        dev_data = self.get_sDevID_data(dev)
        scaled_data = self.preprocessing_data(dev_data)


        # 키워드가 포함된 행 필터링
        host_counts = scaled_data[scaled_data['sHost'].str.contains('|'.join(keyword_lists))].count()[0]

        # 키워드 포함 행이 전체 행중에서 차지하는 비율 
        ratio = round((host_counts / scaled_data.count()[0]) * 100,2)


        # 가장 높은 빈도의 키워드, 빈도, 해당 키워드 비율 
        top_keyword, top_frequency, top_percentage = self.get_top_keyword_frequency(scaled_data, 'sHost', keyword_lists)

        dec_data = self.describe(scaled_data)

        dec_data['카테고리 URL 접속 수'] = host_counts
        dec_data['카테고리 URL 접속 비율(%)'] = ratio
        dec_data['최다 빈도 키워드'] = top_keyword
        dec_data['최다 빈도 키워드 포함 URL 접속 수'] = top_frequency
        dec_data['최다 빈도 키워드 비율(%)'] = top_percentage
        
        return dec_data

    

    # 여러 keyword별 가입자의 데이터를 elasticsearch에 저장 
    def save_keywords_match_data(self, keyword_lists, category_name):
        # 랜덤으로 추출
        dev_id = self.get_keywords_match_devid('sHost', keyword_lists, '10m', random_option=True)
    
        for dev in dev_id:
            try:
                dec_data = self.preprocessing_keywords_match_data(dev, keyword_lists)

                # 해당 카테고리가 전체 URL 접속 비율에서 20% 이상 차지해야 저장
                if dec_data['카테고리 URL 접속 비율(%)'].values < 20:
                    self.logger.info(f"not save {dev} data(index name : {category_name}, < 20%)")
                 
                elif dec_data['접속 시간(분)'].values <= 30 or dec_data['평균 접속 수(1분)'].values <= 1:
                    self.logger.info(f"not save {dev} data(index name : {category_name}, information short)")
                else:
                    self.save_db_data(dec_data, f"{category_name}-describe")
                    self.logger.info(f"save success {dev} data(index name : {category_name})")
                    
            except Exception as e:
                self.logger.error(e)
                continue
    # 2) csv로 저장
        
    def save_csv(self, data, user_id):
        # # 엑셀 저장
        data.tz_localize(None).to_csv('{}_{}.csv'.format(user_id, self.korea_date), encoding='utf-8-sig')

        print('csv 저장 완료 : ',self.korea_date)


    # 3. 전처리 함수

    # 기초 전처리 
    def preprocessing_data(self, data):
        column_matching = ['@timestamp', 'cRes', 'sDevID', 'uDstIp', 'uDstPort', 'nSize', 'sHost', 'sURI', 'sRef','sUA', 'sMwN']
        data = data.loc[:, column_matching]
        data.drop_duplicates(inplace=True)
        
        try:
            data['nSize'] = data['nSize'].astype('int64')
        except: 
            data['nSize'] = data['nSize']
        

        
        data['@timestamp'] = pd.to_datetime(data['@timestamp'])
        # @timestamp 열을 한국 시간으로 변환
        data['@timestamp'] = data['@timestamp'].apply(self.utc_to_kst)
        
        data = data.sort_values('@timestamp')
        
        data['hour'] = data['@timestamp'].dt.hour
        data['minute'] = data['@timestamp'].dt.minute
        
        data['sUA'] = data['sUA'].apply(lambda x : 'No data' if x == '?' else x)
        
        data['timestamp'] = data['@timestamp']
    
        
        data.reset_index(drop=True, inplace=True)
        data.fillna(0, inplace=True)
        
        
        # 데이터프레임의 로그 기록 시간을 인덱스로 설정
        data.set_index('@timestamp', inplace=True)
        
        data.reset_index(inplace=True)
        data['duration'] = data['timestamp'].diff().dt.total_seconds().round(3)
        
        # 이전 행과의 비교를 위해 shift 함수를 사용하여 이전 행의 "sHost" 값을 가져옴
        data['previous_sHost'] = data['sHost'].shift()

        # 이전 행과의 비교 결과가 동일한지 여부를 나타내는 새로운 열 추가
        data['sHost_same_as_previous'] = (data['sHost'] == data['previous_sHost']).astype(int)

        # 데이터프레임에 'sHost_same_as_previous_count' 열 추가
        data['sHost_same_as_previous_count'] = 0
        
        
        # 첫 번째 행은 항상 0으로 설정
        data.loc[data.index[0], 'sHost_same_as_previous_count'] = 0

        # 이전 행의 'sHost_same_as_previous' 값과 현재 행의 'sHost_same_as_previous' 값을 비교하여 count 계산
        for i in range(1, len(data)):
            if data.loc[data.index[i], 'sHost'] == data.loc[data.index[i-1], 'sHost']:
                data.loc[data.index[i], 'sHost_same_as_previous_count'] = data.loc[data.index[i-1], 'sHost_same_as_previous_count'] + 1
            else:
                data.loc[data.index[i], 'sHost_same_as_previous_count'] = 0
        
        data.fillna(0, inplace=True)
        # data = data[1:]
        
        # 데이터프레임의 로그 기록 시간을 인덱스로 설정
        data.set_index('@timestamp', inplace=True)
        
        return data 




    # 인덱스 자동 삭제
    def delete_old_indices(self, index_prefix, num_days_to_keep):

        # 삭제할 기준 날짜 계산
        delete_date = self.korea_time - timedelta(days=num_days_to_keep)
        delete_date_str = delete_date.strftime("%Y-%m-%d")

        # 삭제할 인덱스 이름 리스트 생성
        indices_to_delete = [index for index in self.es.indices.get(index=f"{index_prefix}-*") if index < f"{index_prefix}-{delete_date_str}"]
        
        print(indices_to_delete)
        
        
        # 인덱스 삭제
        for index in indices_to_delete:
            self.es.indices.delete(index=index)



class Modeling(Elk):

    def __init__(self):
        # 현재 날짜와 시간을 가져옴
        now = datetime.now()

        self.current_path = os.getcwd() + "/"

        # 한국 시간대로 변환
        korea_timezone = pytz.timezone("Asia/Seoul")
        korea_time = now.astimezone(korea_timezone)

        # 날짜 문자열 추출
        self.korea_date = korea_time.strftime("%Y-%m-%d")



        self.log_folder = self.current_path + 'logs/model_result/'
        os.makedirs(self.log_folder, exist_ok=True)

        self.log_filename = self.log_folder + f'elastic_program_{self.korea_date}.log'

        # 로깅 핸들러를 생성합니다.
        self.log_handler = logging.FileHandler(self.log_filename)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

        # 로거를 생성하고 로깅 핸들러를 추가합니다.
        self.logger = logging.getLogger('s')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)


        # 변수 선택 
        self.select_columns =  ['평균 접속 수(1분)' ,'평균 접속 간격(초)', '접속 횟수 대비 고유 URL 비율(%)', '최다 이용 UA 접속 비율(%)' ,'최다 연속 URL 접속 비율(%)', '최대 빈도 URL 접속 비율(%)','평균 패킷 길이']

        with open(self.current_path + 'config.yaml', encoding='UTF-8') as f:
            _cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.elasticsearch_ip = _cfg['ELASTICSEARCH_IP_ADDRESS']  

               
        # Elasticsearch 연결
        self.es = Elasticsearch([self.elasticsearch_ip])



    def train_data_preprocessing(self):

#         # 접속 시간이 0인 데이터 삭제 
#         zero_connect_time_index = data[data['접속 시간(분)'] == 0].index
#         data = data.drop(zero_connect_time_index)

#         # 평균 접속 시간이 0인 데이터 삭제 
#         zero_connect_count_index = data[data['평균 접속 수(1분)'] == 0.0].index
#         data = data.drop(zero_connect_count_index)

        data = self.get_index_data('describe*')
                
        data.reset_index(drop=True, inplace=True)
        
        print('학습용 데이터 불러오기 완료')

        data = data[self.select_columns]

        print('모델 전처리 및 학습용 변수 선택 완료')
        print()
        # 표준화 진행 및 저장   
        data = self.save_standard_transform(data)
        print('표준화 완료')
        print()

        return data


    # 데이터 표준화 저장 
    def save_standard_transform(self, data):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        X = data.values

        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        result_data = pd.DataFrame(X_scaled, columns=data.columns)


        # 스케일러 저장 경로
        scaler_path = f'{folder_path}/scaler.pkl'

        # 스케일러 저장
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
            
        print(scaler_path, 'scaler 저장 완료')
        
        return result_data


        
        

    def pca_auto_choice(self, data): 
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

    def pca_num_choice(self, data, num_components): 
        variables = data.columns
        # 현재 폴더 경로 확인
        
        folder_path = self.current_path + "train_models"
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
        pca_model_path = f'{folder_path}/pca_model.pkl'

        # PCA 모델 저장
        with open(pca_model_path, 'wb') as file:
            pickle.dump(pca, file)
            
        print(pca_model_path, 'PCA 저장 완료')

        
        return principalDf 


    def kmeans_modeling(self, k):
        
        
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        # 모델 적합
        print('==== kmeans clustering model 학습을 시작합니다. =====')
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=2023)
        kmeans_model = kmeans.fit(data.values)
        print('모델 학습 중...')
        print()
        
        print(f'kmeans clustering model 학습 완료!(data = {len(data)})')
        print()
        
        # 이상치 군집 저장
        data['kmeans_label'] = kmeans_model.labels_

        print('k 별 count')
        print(data['kmeans_label'].value_counts())
        
        outlier_num = str(data['kmeans_label'].value_counts().index[-1])
        print()
    
        # kmeans 모델 저장 경로
        model_path =  f'{folder_path}/kmeans_model.pkl'
        
    
        outlier_path =  f'{folder_path}/kmeans_outlier_k.yaml'
        
        meta_data = {
            'k': str(k),
            'kmeans_outlier_k': outlier_num
        }

        # 모델 저장
        with open(model_path, 'wb') as file:
            pickle.dump(kmeans_model, file)
        
        print(model_path, '모델 저장 완료')
        
  
        # outlier 번호 저장
        with open(outlier_path, 'w') as file:
            yaml.dump(meta_data, file)


    def GM_modeling(self, n_components=5, threshold=-14):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        # GMM 모델 초기화
        n_components =  n_components # 가우시안 분포 개수
        print(f'n_components = {n_components}')
        
        gmm = GaussianMixture(n_components=n_components)
        
        print('==== GaussianMixure model 학습을 시작합니다. =====')
        
        # 초기 데이터셋으로 모델 학습
        gmm.fit(data.values)
        print('학습 중...')
        print()
        print(f'GaussianMixture Model 학습 완료! data = {len(data)})')
        print()
        
        
        log_probs = gmm.score_samples(data.values)
        threshold = threshold # 임계값 설정 (적절한 값으로 조정해야 함)
        print(f'log likelihood outlier thread : {threshold}')
        
        data['GMM_outlier'] = np.where(log_probs < threshold, 1, 0)
        
        print(data['GMM_outlier'].value_counts()) 
        
        model_path =  f'{folder_path}/gm_model.pkl'
        
        # 모델 저장
        with open(model_path, 'wb') as file:
            pickle.dump(gmm, file)
        
        print(model_path, '모델 저장 완료')
        
        
    def dbscan_modeling(self, eps=1, min_samples=3):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        print('==== DBSCAN clustering model 학습을 시작합니다. =====')
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data.values)
        print('모델 학습 중...')
      
        print()
        print(f'DBSCAN clustering model 학습 완료(eps={eps}, min_samples={min_samples}, data = {len(data)})')
        print()
        data['dbscan_label'] = dbscan.labels_
        outlier_count = data.dbscan_label.value_counts().loc[-1]
        print(f'outlier count : {outlier_count}')
        print(f'outlier ratio : {round((outlier_count / data.shape[0]) * 100, 2) }%')
        
        # kmeans 모델 저장 경로
        model_path =  f'{folder_path}/dbscan_model.pkl'
        
        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(dbscan, f)
        
        print(model_path, '모델 저장 완료')
        
    
#         # Scatter plot 그리기
#         fig = px.scatter(scaled_data, x='component1', y='component2', color=scaled_data['dbscan_label'])


#         # HTML 파일로 저장
#         html_path = self.current_path + "dbscan_scatter_plot.html"
#         fig.write_html(html_path)
        
        
        
#         print(html_path, 'cluster plot 저장 완료')
# #         # 저장된 모델 불러오기
# #         with open('dbscan_model.pkl', 'rb') as f:
# #             loaded_dbscan = pickle.load(f)
    
    def randomforest_modeling(self):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        #dbscan 불러오기
        dm = self.import_dbscan_model()
        
        
        data['dbscan_label'] = dm.labels_
        data['dbscan_label'] = data['dbscan_label'].apply(lambda x : 0 if x != -1 else 1)
        
        
        X = data[self.select_columns]
        
        y = data['dbscan_label']
        
        rf = RandomForestClassifier(random_state=2021)

        
        print('==== Random Forest Classifiy model 학습을 시작합니다. =====')
        rf_model = rf.fit(X, y)
        print('모델 학습 중...')
      
        print()
        print(f'Random Forest Classifiy model 학습 완료! data = {len(data)})')
        print()
     
        # kmeans 모델 저장 경로
        model_path =  f'{folder_path}/rf_model.pkl'
        
        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model , f,  protocol=pickle.HIGHEST_PROTOCOL)
        
        print(model_path, '모델 저장 완료')
        
        
    def isolateforest_modeling(self, contamination = 0.01):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        print('==== Isolation Forest model 학습을 시작합니다. =====')
        isof = IsolationForest(contamination = contamination, random_state=42)
        isof_model = isof.fit(data.values)
        print('모델 학습 중...')
        print()
        
        
        
        print(f'Isolation Forest model  학습 완료(contamination = {contamination}, data = {len(data)})')
        print()
        
        # kmeans 모델 저장 경로
        model_path =  f'{folder_path}/isof_model.pkl'
        
        
        data['isof_label'] = isof_model.predict(data.values)
        print(data['isof_label'].value_counts())
        outlier_count = data.isof_label.value_counts().loc[-1]
        print(f'outlier count : {outlier_count}')
        print(f'outlier ratio : {round((outlier_count / data.shape[0]) * 100, 2) }%')
        
        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(isof_model , f,  protocol=pickle.HIGHEST_PROTOCOL)
        
        print(model_path, '모델 저장 완료')
        
        
    def lof_modeling(self, contamination = 0.01):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # 학습용 데이터 불러오기
        data = self.train_data_preprocessing()
        
        print('==== LOF(Local Outlier Factor) model 학습을 시작합니다. =====')
        lof = LocalOutlierFactor(novelty=True, contamination=contamination) 
        lof_model = lof.fit(data.values)
        print('모델 학습 중...')
        print()
        
        
        print(f'LOF model 학습 완료(contamination = {contamination}, data = {len(data)})')
        print()
        
        # kmeans 모델 저장 경로
        model_path =  f'{folder_path}/lof_model.pkl'
        
        
        data['lof_label'] = lof_model.predict(data.values)
        print(data['lof_label'].value_counts())
        outlier_count = data.lof_label.value_counts().loc[-1]
        print(f'outlier count : {outlier_count}')
        print(f'outlier ratio : {round((outlier_count / data.shape[0]) * 100, 2) }%')
        
        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(lof_model , f,  protocol=pickle.HIGHEST_PROTOCOL)
        
        print(model_path, '모델 저장 완료')
        
    def total_modeling(self):
        self.kmeans_modeling(k=4)
        self.dbscan_modeling(eps=1, min_samples=3)
        self.randomforest_modeling()
        self.isolateforest_modeling()
        self.lof_modeling()
        
        print('모델 업데이트 완료')
        
    
    def import_kmeans_model(self):
        
        
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        
        # kmeans 불러오기
        # 모델 파일 경로
        kmeans_model_path = f'{folder_path}/kmeans_model.pkl'

        # 모델 불러오기
        with open(kmeans_model_path, 'rb') as file:
            kmeans_loaded_model = pickle.load(file)
            
        print(kmeans_model_path,'kmeans import success!')
            

#         # PCA 모델 파일 경로
#         pca_model_path = f'{folder_path}/pca_model.pkl'

#         # PCA 모델 불러오기
#         with open(pca_model_path, 'rb') as file:
#             loaded_pca = pickle.load(file)
            

        return  kmeans_loaded_model
    
    def import_GM_model(self):
        
        
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        
        # kmeans 불러오기
        # 모델 파일 경로
        gm_model_path = f'{folder_path}/gm_model.pkl'

        # 모델 불러오기
        with open(gm_model_path, 'rb') as file:
            gm_loaded_model = pickle.load(file)
            
        print(gm_model_path,'gmm import success!')
            
        return  gm_loaded_model
    
    def import_rf_model(self):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)

        # RandomForest 모델 파일 경로
        rf_model_path = f'{folder_path}/rf_model.pkl'

        # 모델 불러오기
        with open(rf_model_path, 'rb') as file:
            rf_loaded_model = pickle.load(file)

        print(rf_model_path, 'Random Forest Classify Model import success!')

        return rf_loaded_model

    def import_dbscan_model(self):
           
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # dbscan 불러오기
        # 모델 파일 경로
        dbscan_model_path = f'{folder_path}/dbscan_model.pkl'

        # 모델 불러오기
        with open(dbscan_model_path, 'rb') as file:
            dbscan_loaded_model = pickle.load(file)     
        
        print(dbscan_model_path,'dbscan import success!')
        
        return dbscan_loaded_model
    
#       def dbscan_detect_outliers(data):
            
#             # 새로운 데이터를 사용하여 모델 업데이트
#             gmm.fit(np.vstack((initial_data, data)))

#             # 이상치 탐지
#             log_probs = gmm.score_samples(data)
#             threshold = -10.0  # 임계값 설정 (적절한 값으로 조정해야 함)
#             outliers = np.where(log_probs < threshold)[0]

#             return outliers
    def import_isof_model(self):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
      
        # 모델 파일 경로
        isof_model_path = f'{folder_path}/isof_model.pkl'

        # 모델 불러오기
        with open(isof_model_path, 'rb') as file:
            isof_loaded_model = pickle.load(file)     
        
        print(isof_model_path,'Isolation Forest Model import success!')
        
        return isof_loaded_model
    
    def import_lof_model(self):
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
      
        # 모델 파일 경로
        lof_model_path = f'{folder_path}/lof_model.pkl'

        # 모델 불러오기
        with open(lof_model_path, 'rb') as file:
            lof_loaded_model = pickle.load(file)     
        
        print(lof_model_path,'LOF Model import success!')
        
        return lof_loaded_model
        
        
    
    def import_scaler_model(self):
            
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        # 스케일러 파일 경로
        scaler_path = f'{folder_path}/scaler.pkl'

        # 스케일러 불러오기
        with open(scaler_path, 'rb') as file:
            loaded_scaler = pickle.load(file)
        
        print(scaler_path,'scaler import success!')
        
        return loaded_scaler
        
        
    
    
    def gmm_predict(self, data, model):
        log_probs = model.score_samples(data)
        threshold = -15 # 임계값 설정 (적절한 값으로 조정해야 함)
        
        outlier_label = np.where(log_probs < threshold, 1, 0)
        
        return outlier_label
    
    def return_labels(self, data, model_list):
   
        select_data = data[self.select_columns]    
        scaled_data = model_list[-1].transform(select_data.values)
        # kmean
        kmeans_label = model_list[0].predict(scaled_data)[0]
     
        #randomforest
        rf_label =  model_list[1].predict(scaled_data)[0]
    
        # gmm
        gm_label =  self.gmm_predict(scaled_data, model_list[2])[0]
        
        # isolation forest
        isof_label = model_list[3].predict(scaled_data)[0]
        
        
        # lof
        lof_label = model_list[4].predict(scaled_data)[0]
   
        return kmeans_label, rf_label, gm_label, isof_label, lof_label
    
    
    def get_kmeans_outlier_k(self):
        # YAML 파일 경로
        outlier_k_path_kmeans = self.current_path + "train_models/kmeans_outlier_k.yaml"
        
        # YAML 파일 읽기
        with open(outlier_k_path_kmeans, "r") as f:
            yaml_data = yaml.safe_load(f)


        kmeans_outlier_k = int(yaml_data.get("kmeans_outlier_k"))
        
        
        
        return kmeans_outlier_k


    def rule_based_modeling(self, dec_data, dev_id):
        if dec_data['접속 시간(분)'].values <= 30 or dec_data['평균 접속 수(1분)'].values <= 1:
            self.logger.info(f"{dev_id} : information short(접속시간 < 10분 or 평균 접속 수 = 0)")
            return True
        
        if dec_data['최다 접속 URL'].values == 'gms.ahnlab.com' or 'steam' in dec_data['최다 접속 URL'].values:
            self.logger.info(f"{dev_id} : Included in white URL list")
            return True
        
        
        if dec_data["차단 수"].values >= 10 or dec_data["차단율(%)"].values >= 50:
            if dec_data['최대 빈도 URL 접속 비율(%)'].values >= 80 or dec_data['최다 이용 UA 접속 비율(%)'].values >= 80:
                self.logger.info(f"{dev_id} : Anomaly Rule matched!")
                dec_data["판별 등급"] = '위험'
                dec_data["판별 요인"] = 'Rule'
                self.save_db_data(dec_data, "abnormal_describe")
                return True
            else:
                self.logger.info(f"{dev_id} : Anomaly Rule matched!")
                dec_data["판별 등급"] = '의심'
                dec_data["판별 요인"] = 'Rule'
                self.save_db_data(dec_data, "abnormal_describe")
                return True
            
        if dec_data["평균 접속 수(1분)"].values >= 100 and dec_data["최다 이용 UA 접속 비율(%)"].values >= 95 and dec_data["최대 빈도 URL 접속 비율(%)"].values >= 95:
            self.logger.info(f"{dev_id} : Anomaly Rule matched!")
            dec_data["판별 등급"] = '의심'
            dec_data["판별 요인"] = 'Rule'
            self.save_db_data(dec_data, "abnormal_describe")
            return True
        
        if dec_data["최다 접속 URL"].values == "123.57.193.95" or dec_data["최다 접속 URL"].values == "123.57.193.52":
            self.logger.info(f"{dev_id} : Anomaly Rule matched!")
            dec_data["판별 등급"] = '위험'
            dec_data["판별 요인"] = 'Rule'
            self.save_db_data(dec_data, "abnormal_describe")
            return True
        return False


    def return_total_label_to_elasticsearch(self, dev_id, model_list):
        
        # kmeams outlier 
        kmeans_outlier_k = self.get_kmeans_outlier_k()
        
        
        dec_data = self.get_final_dec_data_dev_id(dev_id)    
        
        # 현재 시간 구하기
        now = datetime.now()

        # 한국 시간대로 변환
        korea_timezone = pytz.timezone("Asia/Seoul")
        korea_time = now.astimezone(korea_timezone)


        dec_data['timestamp'] = korea_time
        
        
        
        # rule based model 
        rule_yes = self.rule_based_modeling(dec_data, dev_id)
        
        if rule_yes == True:
            return
        # kmeans 
        kmeans_label, rf_label, gm_label, isof_label, lof_label = self.return_labels(dec_data, model_list)
        
        outlier_count = 0
        
        if kmeans_label == kmeans_outlier_k:
            outlier_count += 1
            
        if rf_label == 1:
            outlier_count += 1
        if gm_label == 1:
            outlier_count += 1
        
        if isof_label == -1:
            outlier_count += 1
        
        if lof_label == -1:
            outlier_count += 1
            
        if outlier_count >= 4:
            self.logger.info(f"{dev_id}(km_label = {kmeans_label}, rf_label = {rf_label}, gm_label = {gm_label}, isof_label = {isof_label}, lof_label = {lof_label},detect_count = {outlier_count}) : Model Detection matched!")
            dec_data["판별 등급"] = '위험'
            dec_data["판별 요인"] = 'ML Model'
            self.save_db_data(dec_data, "abnormal_describe")
            return  
        
        elif outlier_count >= 3:
            self.logger.info(f"{dev_id}(km_label = {kmeans_label}, rf_label = {rf_label}, gm_label = {gm_label}, isof_label = {isof_label}, lof_label = {lof_label}, detect_count = {outlier_count}) : Model Detection matched!")
            dec_data["판별 등급"] = '의심'
            dec_data["판별 요인"] = 'ML Model'
            self.save_db_data(dec_data, "abnormal_describe")
            return  
    
        self.logger.info(f"{dev_id}(km_label = {kmeans_label}, rf_label = {rf_label}, gm_label = {gm_label}, isof_label = {isof_label}, lof_label = {lof_label}) : Normal User")
               
    def total_model_load(self):
        kmeans_loaded_model = self.import_kmeans_model()
        gm_loaded_model = self.import_GM_model()
        rf_loaded_model = self.import_rf_model()
        isof_loaded_model = self.import_isof_model()
        lof_loaded_model = self.import_lof_model()
        loaded_scaler  = self.import_scaler_model()
        
        
        model_list = [kmeans_loaded_model, rf_loaded_model, gm_loaded_model, isof_loaded_model, lof_loaded_model,loaded_scaler]
    
        return model_list

    def process(self):
        # 랜덤 샘플링 버전
        # dev_id_list = get_sDevID_random("1m")

        # 허용 block 혼합 버전
        dev_id_list = self.get_pass_block_dev_list()
        
        # 모델 로드
        model_list = self.total_model_load()
        

        for dev_id in dev_id_list:
            self.return_total_label_to_elasticsearch(dev_id, model_list)

# 데이터를 로드하고 전처리한 후 X에 저장
# 최대 클러스터 개수 설정
    def clustering_choice_k_scree(self, data, max_clusters):
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

        return plt


    def visualize_sil(self, data, max_clusters):
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
        return plt
    


    def get_device_id_dash(self, dev_id):
        dev_data = self.get_sDevID_data(dev_id)
        data = self.preprocessing_data(dev_data)
        describe_data = self.describe(data)
    
        
        user_name = describe_data['가입자 ID'][0]
        ip_describe = self.get_ip_describe(data)
        top_country = ip_describe.groupby('country')['connect_count'].sum().sort_values(ascending=False)[:1].index[0]
        
        
        specs = [
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'xy', 'colspan': 2, 'rowspan': 2}, None, {'type': 'xy', 'colspan': 2, 'rowspan': 2}, None],
            [None, None, None, None],
            [{'type': 'xy', 'colspan': 2, 'rowspan': 2}, None, {'type': 'xy', 'colspan': 2, 'rowspan': 2}, None],
            [None, None, None, None],
            [{'type': 'indicator', 'colspan' : 4, 'rowspan': 2}, None, None, None], 
            [None, None, None, None],     
        ]
        
        font_family = 'Pretendard Black, sans-serif'

        from elk.visualize import Visualize

        v = Visualize()

        # {'type': 'xy', 'colspan': 2}, None,
        fig1 = v.gage_chart(describe_data, '차단율(%)')
        fig2 = v.gage_chart(describe_data, '최대 빈도 URL 접속 비율(%)')
        fig3 = v.gage_chart(describe_data, '최다 이용 UA 접속 비율(%)')
        fig4 = v.gage_chart(describe_data, '접속 횟수 대비 고유 URL 비율(%)')
        fig6 = v.text_chart(describe_data, '전체 접속 횟수')
        fig7 = v.text_chart(describe_data, '평균 접속 수(1분)')
        fig8 = v.text_chart(describe_data, '평균 접속 간격(초)')
        fig9 = v.text_chart(describe_data, '접속 UA 수')
        fig10 = v.value_counts_top10_bar(data, 'sHost')
        fig11 = v.value_counts_top10_bar(data, 'sUA')
        fig12 = v.seasonal_decompose_plot(data, period=5)
        fig13 = v.ip_location_table(ip_describe)

        # 대시보드 그래프 배열
        fig = make_subplots(
            rows=8, cols=4,
            vertical_spacing=0.1,
            horizontal_spacing=0.1, 
            specs=specs,  # 그래프 간의 수직 간격 조정
            subplot_titles=['', '', '', '', 
                        '', '','','',
                        'Connect Pattern','Top 10 URL/UA', '', '', 
                            f"Country of the IP address with the highest proportion : {top_country}"]
        )

        fig10.data[0]['showlegend'] = False
        fig11.data[0]['showlegend'] = False
        fig12.data[0]['showlegend'] = False
        fig12.data[2]['showlegend'] = False

        fig10.data[0]['marker']['color'] = 'rgb(211, 211, 211)'
        fig11.data[0]['marker']['color'] = 'rgb(211, 211, 211)'

        # 그래프에 폰트 적용
        fig.update_layout(
            font=dict(family=font_family)
        )

        fig.add_trace(fig1.data[0], row=2, col=1)
        fig.add_trace(fig2.data[0], row=2, col=2)
        fig.add_trace(fig3.data[0], row=2, col=3)
        fig.add_trace(fig4.data[0], row=2, col=4)
        # fig.add_trace(fig5.data[0], row=1, col=1)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig6.data[0], row=1, col=1)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig7.data[0], row=1, col=2)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig8.data[0], row=1, col=3)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig9.data[0], row=1, col=4)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig10.data[0], row=3, col=3)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig11.data[0], row=5, col=3)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig12.data[0], row=3, col=1)
        fig.add_trace(fig12.data[2], row=5, col=1)
        fig.add_trace(fig13.data[0], row=7, col=1)

        # fig.add_trace(fig12.data[1], row=6, col=1)
        # fig.add_trace(fig12.data[2], row=7, col=1)
        # fig.add_trace(fig12.data[3], row=8, col=1)


        # 특정 그래프의 Y 축 제목 설정
        fig.update_yaxes(title_text='Original', title_font=dict(size=15), row=3, col=1)
        fig.update_yaxes(title_text='Seasonality', title_font=dict(size=15), row=5, col=1)


        # 표 레이아웃 설정
        fig.update_layout(
            title={
                'text': f"<{user_name}> Status in the Last 1 Hour",
                'font': {'size': 30, 'family': font_family}
            },
        )



        # # 대시보드 출력
        # fig.show()

        # 그래프를 HTML로 저장
        # html_path = self.current_path + 'dashboard.html'
        # pyo.plot(fig, filename= html_path)
        # print(html_path, ' dashboard 저장 완료!')
        # print()
        return fig 


# app = SEGMENT()

# print(app.get_sDevID_random('10m'))