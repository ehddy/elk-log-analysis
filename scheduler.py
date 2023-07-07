import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import time
import logging
import pandas as pd
import schedule

def get_data():

    redis_client = redis.Redis(host='172.17.0.2', port=6379)
    data_list = []
    for key in redis_client.keys():
        data_list.append((key.decode(), redis_client.get(key).decode()))
    return data_list


def insert_data():

    # 현재 시간을 나타내는 문자열을 생성 
    time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # 로그 파일 이름에 현재 시간을 포함
    log_filename = f'/data/logs/save_{time_str}.log'

    # 로깅 핸들러를 생성
    log_handler = logging.FileHandler(log_filename)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    # 로거를 생성하고 로깅 핸들러를 추가합니다.
    logger = logging.getLogger(f'{time_str}')
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    conn_info = {
    "host": "172.17.0.3",
    "database": "plantynet",
    "port":"5432",
    "user": "ehddy",
    "password": "4631"
    }

    # PostgreSQL 연결
    try:
        conn = psycopg2.connect(**conn_info)
        data_list = get_data()
        cursor = conn.cursor()
        for data in data_list:
            cursor.execute("""
                INSERT INTO select_table(url, count)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
            """, (data[0], int(data[1])))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f'success save {len(data_list)} data')
    except:
        logger.warn("Empty redis data(error)")
        


# 3분에 한 번씩 insert_data() 함수 실행
schedule.every(60).minutes.do(insert_data)

while True:
    schedule.run_pending()
    time.sleep(1)
