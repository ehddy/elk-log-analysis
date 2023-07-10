from segment import *

# 2일 이상이 된, 비정상 접속 의심 가입자 인덱스 삭제
delete_old_indices('abnormal_describe', 2)