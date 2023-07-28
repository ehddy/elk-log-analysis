from elk_program import Modeling
import schedule
import time

def run_scheduler():
    model = Modeling()
    model.process()

    schedule.every(0.5).minutes.do(model.process)
    while True:
        schedule.run_pending()
        time.sleep(1)

# 스케줄러를 실행
run_scheduler()