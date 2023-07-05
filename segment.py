
from elasticsearch import Elasticsearch

import pandas as pd

import logging

from datetime import datetime
import pytz

from cluster import *
from graph import *

import yaml

import requests
import json
import socket

import plotly.figure_factory as ff


import numpy as np
import plotly.graph_objects as go

from datetime import timedelta

import random

import plotly.express as px

import plotly.subplots as sp
import statsmodels.api as sm


import warnings
warnings.filterwarnings('ignore')


with open('config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

elasticsearch_ip = _cfg['ELASTICSEARCH_IP_ADDRESS']    
    
    
# UTC 시간을 한국 시간으로 변환하는 함수
def utc_to_kst(utc_time):
    kst = pytz.timezone('Asia/Seoul')
    kst_time = utc_time.astimezone(kst)
    return kst_time



# 1. get(데이터 수집 관련 함수)

# 1) 가입자 리스트 추출 

# 랜덤하게 가입자 추출
def get_sDevID_random():
    # Elasticsearch 연결
    es = Elasticsearch([elasticsearch_ip])

    # Elasticsearch에서 데이터 검색
    res = es.search(
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
                        "gte": "now-1h",
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
    random_sDevID_list = random.sample(sDevID_list, 500)


    return random_sDevID_list

# 차단 허용 기준으로 가입자 추출
def get_sDevID(size, cRes):
    # Elasticsearch 연결
    es = Elasticsearch([elasticsearch_ip])

    # Elasticsearch에서 데이터 검색
    res = es.search(
        index='sa*',
        body={
        "size": 0,
        "aggs": {
            "devid_count": {
                "terms": {
                    "field": "sDevID.keyword",
                    "min_doc_count": 1000,
                    "size": size,
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
                                "gte": "now-1h",
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

# 키워드 기준으로 가입자 추출(100명)
def get_keywords_match_devid_100(column_name, match_keywords):
    es = Elasticsearch([elasticsearch_ip])
    
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
    res = es.search(
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
def get_index_data(index_name):

    # Elasticsearch 연결
    es = Elasticsearch([elasticsearch_ip])

    # 가져올 인덱스 이름 설정
    index_name = '{}*'.format(index_name)

    # Elasticsearch에서 모든 데이터 검색
    res = es.search(index=index_name, body={"query": {"match_all": {}}}, size=10000, scroll='10m')


    data = pd.DataFrame([hit['_source'] for hit in res['hits']['hits']])
    
    
    scroll_id = res['_scroll_id']
    hits_list = []
    
    while True:
        res = es.scroll(scroll_id=scroll_id, scroll='10m')
        if len(res['hits']['hits']) == 0:
            break
        hits = res['hits']['hits']
        hits_list.extend(hits)
    
    es.clear_scroll(scroll_id=scroll_id)
    data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
    data = pd.concat([data, data_scroll])

    
    return data
    
    return df 


# 가입자 이름을 기준으로 모든 데이터 추출 
def get_sDevID_data(user_id):
    es = Elasticsearch([elasticsearch_ip])
    # Elasticsearch에서 데이터 검색
    res = es.search(
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
        res = es.scroll(scroll_id=scroll_id, scroll='10m')
        if len(res['hits']['hits']) == 0:
            break
        hits = res['hits']['hits']
        hits_list.extend(hits)
    
    es.clear_scroll(scroll_id=scroll_id)
    data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
    data = pd.concat([data, data_scroll])

    
    return data


# 특정 키워드 1개를 기준으로 포함 데이터 가져오기
def get_category_match_data(column_name, match_keyword):
    es = Elasticsearch([elasticsearch_ip])
    # Elasticsearch에서 데이터 검색
    res = es.search(
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
        res = es.scroll(scroll_id=scroll_id, scroll='10m')
        if len(res['hits']['hits']) == 0:
            break
        hits = res['hits']['hits']
        hits_list.extend(hits)
    
    es.clear_scroll(scroll_id=scroll_id)
    data_scroll = pd.DataFrame([hit['_source'] for hit in hits_list])
    data = pd.concat([data, data_scroll])

    return data


# 3) 데이터 통합용 

# 허용, 차단을 기준으로 최종 데이터 추출 (szie는 추출하고 싶은 가입자의 수)
def get_data(size, cRes):
    sDevID_list = get_sDevID(size, cRes)  
    total_df_list = []
    for dev_id in sDevID_list:
        data = get_sDevID_data(dev_id)
        scaled_data = preprocessing_data(data)
        new_data = add_new_columns(scaled_data)
        total_df_list.append(new_data)
        save_csv(new_data, dev_id)
    
    return total_df_list

# 모델 통합용(전처리 통합)
def get_final_dec_data_dev_id(dev_id):
    dev_data = get_sDevID_data(dev_id)
    scaled_data = preprocessing_data(dev_data)
    new_data = add_new_columns(scaled_data)
    dec_data = describe(new_data)
    
    return dec_data


# 4) 집계용 

# 1시간 동안의 유저 데이터를 요약 집계 
def describe(data):
    
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
def describe_category(data, column_name):
    unique_list = list(data[column_name].unique())
    categoty_segment_data = []
    dev_id = data['sDevID'][0]
    unique_describe = []
    for name in unique_list:
        category_data = data.loc[data[column_name] == name, :]
        new_category_data = add_new_columns(category_data)
        categoty_segment_data.append(new_category_data)
        dec = describe(new_category_data)
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
def get_top_keyword_frequency(data, column_name, keywords):
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
def get_geolocation(ip_address):
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
def get_ip_describe(data):
    host_list = data.sHost.value_counts()[:10].index
    describe_url_df = data.groupby(['sHost', 'uDstIp', 'sUA'])[['cRes']].count().reset_index().rename({'cRes' : 'connect_count'}, axis=1)
    describe_url_df = describe_url_df[describe_url_df['sHost'].isin(host_list)].sort_values('connect_count', ascending=False).reset_index(drop=True)
    ip_list  = describe_url_df['uDstIp'].values
    ip_location_list = []
    for ip in ip_list:
        location = get_geolocation(ip)
        ip_location_list.append(location)

    describe_url_df[['country', 'regionName', 'city']] = ip_location_list
    
    return describe_url_df



## 2. save(저장)


# 1) Elasticsearch에 저장 

# 랜덤 가입자 집계 데이터 저장 
def save_db_random_devid():
    es = Elasticsearch([elasticsearch_ip])
    
    
    # 현재 날짜와 시간을 가져옴
    now = datetime.now()

    # 한국 시간대로 변환
    korea_timezone = pytz.timezone("Asia/Seoul")
    korea_time = now.astimezone(korea_timezone)

    # 날짜 문자열 추출
    korea_date = korea_time.strftime("%Y-%m-%d")

    sDevID_list = get_sDevID_random()
    for dev_id in sDevID_list:
        
        data = get_sDevID_data(dev_id)
        scaled_data = preprocessing_data(data)
        new_data = add_new_columns(scaled_data)
        dec_data = describe(new_data)
        # 데이터프레임을 Elasticsearch 문서 형식으로 변환
        documents = dec_data.to_dict(orient='records')

        # Elasticsearch에 데이터 색인
        index_name = f'describe-{korea_date}'  # 저장할 인덱스 이름
        for doc in documents:
            es.index(index=index_name, body=doc)
    print('{}개 저장 완료'.format(len(sDevID_list)))
    
    
# 집계 데이터 특정 index_name에 저장 
def save_db_data(dec_data, index_name):
    es = Elasticsearch([elasticsearch_ip])
    
    # 현재 날짜와 시간을 가져옴
    now = datetime.now()

    # 한국 시간대로 변환
    korea_timezone = pytz.timezone("Asia/Seoul")
    korea_time = now.astimezone(korea_timezone)

    # 날짜 문자열 추출
    korea_date = korea_time.strftime("%Y-%m-%d")

    # 데이터프레임을 Elasticsearch 문서 형식으로 변환
    documents = dec_data.to_dict(orient='records')

    # Elasticsearch에 데이터 색인
    index_name = f'{index_name}-{korea_date}'  # 저장할 인덱스 이름
    for doc in documents:
        es.index(index=index_name, body=doc)
        

# 여러 keyword별 top 100 가입자의 데이터를 elasticsearch에 저장 
def save_keywords_match_data(keyword_lists, category_name):
    dev_id = get_keywords_match_devid_100('sHost', keyword_lists)
    for dev in dev_id:
        dev_data = get_sDevID_data(dev)
        dev_data = preprocessing_data(dev_data)
        new_data = add_new_columns(dev_data)

        
        # 키워드가 포함된 행 필터링
        host_counts = new_data[new_data['sHost'].str.contains('|'.join(keyword_lists))].count()[0]
        
        # 키워드 포함 행이 전체 행중에서 차지하는 비율 
        ratio = round((host_counts / new_data.count()[0]) * 100,2)
        
        
        # 가장 높은 빈도의 키워드, 빈도, 해당 키워드 비율 
        top_keyword, top_frequency, top_percentage = get_top_keyword_frequency(new_data, 'sHost', keyword_lists)
        
        dec_data = describe(new_data)
        
        dec_data['카테고리'] = category_name
        dec_data['카테고리 URL 접속 수'] = host_counts
        dec_data['카테고리 URL 접속 비율(%)'] = ratio
        dec_data['최다 빈도 키워드'] = top_keyword
        dec_data['최다 빈도 키워드 포함 URL 접속 수'] = top_frequency
        dec_data['최다 빈도 키워드 비율(%)'] = top_percentage
        
        
        save_db_data(dec_data, f"{category_name}-describe")

# 2) csv로 저장
    
def save_csv(data, user_id):
    # 현재 날짜와 시간을 가져옴
    now = datetime.now()

    # 한국 시간대로 변환
    korea_timezone = pytz.timezone("Asia/Seoul")
    korea_time = now.astimezone(korea_timezone)

    # 날짜 문자열 추출
    korea_date = korea_time.strftime("%Y-%m-%d %H:%M")


    # # 엑셀 저장
    data.tz_localize(None).to_csv('{}_{}.csv'.format(user_id, korea_date), encoding='utf-8-sig')

    print('csv 저장 완료 : ',korea_date)


# 3. 전처리 함수

# 기초 전처리 
def preprocessing_data(data):
    column_matching = ['@timestamp', 'cRes', 'sDevID', 'uDstIp', 'uDstPort', 'nSize', 'sHost', 'sURI', 'sRef','sUA', 'sMwN']
    data = data.loc[:, column_matching]
    data.drop_duplicates(inplace=True)
    
    try:
        data['nSize'] = data['nSize'].astype('int64')
    except: 
        data['nSize'] = data['nSize']
    

    
    data['@timestamp'] = pd.to_datetime(data['@timestamp'])
    # @timestamp 열을 한국 시간으로 변환
    data['@timestamp'] = data['@timestamp'].apply(utc_to_kst)
    
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
def add_new_columns(data):
    
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
    

