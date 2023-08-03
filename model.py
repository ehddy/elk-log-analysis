from elk_program import Modeling
import schedule
import time

def start(model):
    model.process()
    model.cleanup()

def run_scheduler():
    model = Modeling()

    schedule.every(0.5).minutes.do(start, model)  # start 함수를 호출할 때 model 인스턴스를 전달
    while True:
        schedule.run_pending()
        time.sleep(1)

# 스케줄러를 실행
run_scheduler()