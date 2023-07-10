from segment import *
import schedule
import time


def run_scheduler():
    save_db_random_devid()

    schedule.every(10).minutes.do(save_db_random_devid)

    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()