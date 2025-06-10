import shutil

# 예측 결과 폴더 강제 초기화
shutil.rmtree('runs/detect/predict/test', ignore_errors=True)
