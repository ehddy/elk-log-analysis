from elk_program import Modeling
import schedule
import time

def model_start():
    model = Modeling()
    model.process()

def run_scheduler():
    model_start()

    schedule.every(0.5).minutes.do(model_start)

    while True:
        schedule.run_pending()
        time.sleep(1)

# 스케줄러를 실행
run_scheduler()