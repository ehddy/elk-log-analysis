from elk_program import Elk
import schedule
import time



def save_train_data():
    app = Elk()
    app.save_db_random_devid()

def run_scheduler():
    save_train_data()

    schedule.every(1).minutes.do(save_train_data)

    while True:
        schedule.run_pending()
        time.sleep(1)


# 스케줄러를 실행
run_scheduler()



