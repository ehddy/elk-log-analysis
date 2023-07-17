# 3일 이상된 로그 파일 삭제
find /code/logs/save_elasticsearch/ -type f -name "elastic_program_*.log" -mtime +1 -exec rm {} \;
find /code/logs/model_result/ -type f -name "elastic_program_*.log" -mtime +1 -exec rm {} \;

