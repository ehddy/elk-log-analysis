from elk_program import *

# 2일 이상이 된, 비정상 접속 의심 가입자 인덱스 삭제


app = Elk()

app.delete_old_indices('abnormal_describe', 1)