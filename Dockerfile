FROM python:3.11-slim

WORKDIR /app

# ffmpeg는 시스템 패키지이므로 apt로 설치 필요
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# pip 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 소스 복사
COPY . .

# 실행 (필요 시 main.py 또는 main:app 조정)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
