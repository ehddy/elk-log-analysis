from elk_program import Elk
import schedule
import time


def run_scheduler():
    app = Elk()
    app.save_db_random_devid()

    schedule.every(10).minutes.do(app.save_db_random_devid)

    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()