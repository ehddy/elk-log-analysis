from elk_program import Elk
import schedule
import time

def start(app):
    app.save_db_random_devid()
    app.cleanup()


def run_scheduler():
    app = Elk()


    schedule.every(0.5).minutes.do(start, app)
    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()



