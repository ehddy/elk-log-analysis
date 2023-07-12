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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import statsmodels.api as sm

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
        korea_time = now.astimezone(korea_timezone)

        # 날짜 문자열 추출
        self.korea_date = korea_time.strftime("%Y-%m-%d")


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
        pass_list = self.get_sDevID('허용', 5000, '30m')
        
        total_dev_list = list(set(block_list + pass_list))
        
        return total_dev_list

    # 키워드 기준으로 가입자 추출(100명)
    def get_keywords_match_devid_100(self, column_name, match_keywords): 
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
                                        "gte": "now-1h",
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": 0,  # 집계된 결과만 가져옴
                "aggs": {
                    "top_100_devid": {
                        "terms": {
                            "field": "sDevID.keyword",
                            "size": 100,
                            "order": {"_count": "desc"}
                        }
                    }
                }
            }
        )
        
        # 상위 50개의 sDevID 추출
        buckets = res["aggregations"]["top_100_devid"]["buckets"]
        top_100_devid_list = [bucket["key"] for bucket in buckets]
        
        return top_100_devid_list

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
                                    column_name : match_keyword  # 해당 유저의 아이디로 필터링
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
        new_data = self.add_new_columns(scaled_data)
        dec_data = self.describe(new_data)
        
        return dec_data

    # 3) 데이터 통합용 

    # 허용, 차단을 기준으로 최종 데이터 추출 (szie는 추출하고 싶은 가입자의 수)
    def get_data(self, size, cRes):
        sDevID_list = self.get_sDevID(size, cRes)  
        total_df_list = []
        for dev_id in sDevID_list:
            data = self.get_sDevID_data(dev_id)
            scaled_data = self.preprocessing_data(data)
            new_data = self.add_new_columns(scaled_data)
            total_df_list.append(new_data)
            self.save_csv(new_data, dev_id)
        
        return total_df_list

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

        duration_min = round(data['duration'].min(), 2)
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
        decribe_list = [data['sDevID'][0],connect_count,  mean_connect_url_count, block_count,  block_ratio, duration_mean, duration_max, duration_min, 
                        connect_url_count, succession_url_connect_count, main_url,
                        max_frequent_url_connect_count,connect_ua_count, main_ua, main_ua_connect_count
                        ,connect_date, connect_duration, connect_minutes, 
                    mean_packet_lenth,  main_port]
        
        
        column_lists = ['가입자 ID','전체 접속 횟수', '평균 접속 수(1분)','차단 수', '차단율(%)','평균 접속 간격(초)', '최장 접속 간격(초)', '최단 접속 간격(초)', '고유 접속 URL 수', '최대 연속 URL 접속 횟수', 
                    '최다 접속 URL', '최대 빈도 URL 접속 횟수' ,'접속 UA 수', '최다 이용 UA', 
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
            new_category_data = self.add_new_columns(category_data)
            categoty_segment_data.append(new_category_data)
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
        sDevID_list = self.get_sDevID_random("1m")
        for dev_id in sDevID_list:
            try:
                dec_data = self.get_final_dec_data_dev_id(dev_id)
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
            

    # 여러 keyword별 top 100 가입자의 데이터를 elasticsearch에 저장 
    def save_keywords_match_data(self, keyword_lists, category_name):
        dev_id = self.get_keywords_match_devid_100('sHost', keyword_lists)
        for dev in dev_id:
            dev_data = self.get_sDevID_data(dev)
            dev_data = self.preprocessing_data(dev_data)
            new_data = self.add_new_columns(dev_data)

            
            # 키워드가 포함된 행 필터링
            host_counts = new_data[new_data['sHost'].str.contains('|'.join(keyword_lists))].count()[0]
            
            # 키워드 포함 행이 전체 행중에서 차지하는 비율 
            ratio = round((host_counts / new_data.count()[0]) * 100,2)
            
            
            # 가장 높은 빈도의 키워드, 빈도, 해당 키워드 비율 
            top_keyword, top_frequency, top_percentage = get_top_keyword_frequency(new_data, 'sHost', keyword_lists)
            
            dec_data = self.describe(new_data)
            
            dec_data['카테고리'] = category_name
            dec_data['카테고리 URL 접속 수'] = host_counts
            dec_data['카테고리 URL 접속 비율(%)'] = ratio
            dec_data['최다 빈도 키워드'] = top_keyword
            dec_data['최다 빈도 키워드 포함 URL 접속 수'] = top_frequency
            dec_data['최다 빈도 키워드 비율(%)'] = top_percentage
            
            
            self.save_db_data(dec_data, f"{category_name}-describe")

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
        
        return data 



    # 새로운 열 추가     
    def add_new_columns(self, data):
        
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
        self.select_columns = ['평균 접속 수(1분)', '최다 이용 UA 접속 수', '최대 빈도 URL 접속 횟수', '평균 접속 횟수(1개 URL)', '최대 연속 URL 접속 횟수', '고유 접속 URL 수', '평균 패킷 길이']

        with open(self.current_path + 'config.yaml', encoding='UTF-8') as f:
            _cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.elasticsearch_ip = _cfg['ELASTICSEARCH_IP_ADDRESS']  

               
        # Elasticsearch 연결
        self.es = Elasticsearch([self.elasticsearch_ip])



    def train_data_preprocessing(self, data):
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
    def standard_transfrom(self, data):
        X  = data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
        
        

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
        data = self.get_index_data('describe*').reset_index(drop=True)
        
        # 학습용 데이터 전처리(중복 데이터 삭제, 접속시간=0, 접속수=0 데이터 삭제)
        data = self.train_data_preprocessing(data)
        

        data = data[self.select_columns]
        
        # pca 진행
        principalDf = self.pca_num_choice(data, 2)
        
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
        outlier_k_path = self.current_path + "train_models/kmeans_outlier_k.yaml"

        
        
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
        model_path =  f'{folder_path}/kmeans_model.pkl'
        
        

        # 모델 저장
        with open(model_path, 'wb') as file:
            pickle.dump(kmeans_model, file)
        
        print(model_path, '모델 저장 완료')
        # Scatter plot 그리기
        fig = px.scatter(principalDf, x='component1', y='component2', color=principalDf['kmeans_label'])



        # HTML 파일로 저장
        html_path = self.current_path + "kmeans_scatter_plot.html"
        fig.write_html(html_path)
        print(html_path, 'cluster plot 저장 완료')

#     # 스케일러 저장 경로
#     scaler_path = current_path  + f'{folder_path}/scaler.pkl'

#     # 스케일러 저장
#     with open(scaler_path, 'wb') as file:
#         pickle.dump(scaler, file)

    def import_model(self):
        
        
        folder_path = self.current_path + "train_models"
        os.makedirs(folder_path, exist_ok=True)
        
        
        # kmeans 불러오기
        # 모델 파일 경로
        kmeans_model_path = f'{folder_path}/kmeans_model.pkl'

        # 모델 불러오기
        with open(kmeans_model_path, 'rb') as file:
            kmeans_loaded_model = pickle.load(file)

        # PCA 모델 파일 경로
        pca_model_path = f'{folder_path}/pca_model.pkl'

        # PCA 모델 불러오기
        with open(pca_model_path, 'rb') as file:
            loaded_pca = pickle.load(file)
            
    #     # 스케일러 파일 경로
    #     scaler_path = current_path + f'{folder_path}/scaler.pkl'

    #     # 스케일러 불러오기
    #     with open(scaler_path, 'rb') as file:
    #         loaded_scaler = pickle.load(file)
        return  kmeans_loaded_model, loaded_pca
    
    
    def kmeans_predict(self, data, model):
        # # 주성분으로 이루어진 데이터 프레임 구성
        cluster_label = model.predict(data.values)
        
        return cluster_label
    
    def return_labels(self, data):
        kmeans_loaded_model, loaded_pca  = self.import_model()
        

        select_data = data[self.select_columns]
        
        
        printcipalComponents = loaded_pca.transform(select_data)

        principalDf = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2'])
        

        # kmean
        kmeans_label = self.kmeans_predict(principalDf, kmeans_loaded_model)
        #dbscan_label = 
        
        return kmeans_label
    
    def get_kmeans_outlier_k(self):
        # YAML 파일 경로
        outlier_k_path_kmeans = self.current_path + "train_models/kmeans_outlier_k.yaml"
        
        # YAML 파일 읽기
        with open(outlier_k_path_kmeans, "r") as f:
            yaml_data = yaml.safe_load(f)


        kmeans_outlier_k = int(yaml_data.get("kmeans_outlier_k"))
        
        
        
        return kmeans_outlier_k


    def rule_based_modeling(self, dec_data, dev_id):
        if dec_data["평균 접속 수(1분)"].values >= 100 and dec_data["차단 수"].values >= 50 and dec_data["최다 이용 UA 접속 비율(%)"].values >= 90 and dec_data["최대 빈도 URL 접속 비율(%)"].values >= 90:
            self.logger.info(f"{dev_id} : Rule 1 matched!")
            self.save_db_data(dec_data, "abnormal_describe")
            return 
        elif dec_data["최다 접속 URL"].values == "123.57.193.95" or dec_data["최다 접속 URL"].values == "123.57.193.52":
            self.logger.info(f"{dev_id} : Rule 2 matched!")
            self.save_db_data(dec_data, "abnormal_describe")
            return


    def return_total_label_to_elasticsearch(self, dev_id):
        
        # kmeams outlier 
        kmeans_outlier_k = self.get_kmeans_outlier_k()
        
        
        dbscan_outlier_k = -1
        
        dec_data = self.get_final_dec_data_dev_id(dev_id)    
        
        # rule based model 
        self.rule_based_modeling(dec_data, dev_id)
        
        
        # kmeans 
        kmeans_label = self.return_labels(dec_data)[0]
        
        
        if kmeans_label == kmeans_outlier_k:
            self.logger.info(f"{dev_id}(label = {kmeans_outlier_k}) : Rule 3 matched!(kmeans)")
            self.save_db_data(dec_data, "abnormal_describe")
            return 
        # dbscn
        self.logger.info(f"{dev_id}(label = {kmeans_label}) : Normal User")
               

    def process(self):
        # 랜덤 샘플링 버전
        # dev_id_list = get_sDevID_random("1m")

        # 허용 block 혼합 버전
        dev_id_list = self.get_pass_block_dev_list()


        for dev_id in dev_id_list:
            self.return_total_label_to_elasticsearch(dev_id)

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
        scaled_data = self.preprocessing_data(dev_data)
        data = self.add_new_columns(scaled_data)
        describe_data = self.describe(data)
        return_label = self.return_labels(describe_data)
        labels_df = pd.DataFrame(return_label, columns=["label"])
        
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
            [{'type': 'indicator', 'colspan' : 2, 'rowspan': 2}, None, {'type': 'indicator', 'colspan' : 2, 'rowspan': 2}, None], 
            [None, None, None, None],     
        ]
        
        font_family = 'Pretendard Black, sans-serif'

        from visualize import Visualize

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
        fig14 = v.text_chart(labels_df, 'label')

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
        fig.add_trace(fig14.data[0], row=7, col=3)

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