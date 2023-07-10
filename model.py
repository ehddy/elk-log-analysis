from cluster import *
import schedule
import time



def run_scheduler():
    process()

    # 1시간마다 get_db() 함수를 실행하는 스케줄러 함수입니다.
    schedule.every(10).minutes.do(process)

    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()