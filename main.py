from elk_db import get_db
import schedule
import time

def run_scheduler():
    get_db()

    # 1시간마다 get_db() 함수를 실행하는 스케줄러 함수입니다.
    schedule.every(0.5).hours.do(get_db)

    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()