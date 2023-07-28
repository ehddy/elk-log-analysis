from elk_program import Elk
import schedule
import time


shopping_keywords = ['11st', 'auction', 'coupang', 'danawa', 'cjonstyle', 'e-highmart', 'emart', 'ssg', 'gmarket', 'gsshop', 
                        'interpark', 'aliexpress', 'lotteimall', 'qoo10', 'shopping', 'smartstore', 'tmon', 'shop', 'wadiz', 'store']



def run_scheduler():
    app = Elk()
    app.save_keywords_match_data(shopping_keywords, 'shopping')

    schedule.every(1).minutes.do(app.save_keywords_match_data(shopping_keywords, 'shopping'))
    
    app = Elk()
    while True:
        schedule.run_pending()
        time.sleep(1)

    
# 스케줄러를 실행
run_scheduler()     