# 1) conda 베이스 이미지 선택
FROM continuumio/miniconda3:latest

# 2) 환경 파일 복사 및 conda env 생성
#    만약 environment.vml 이 실제론 environment.yml 포맷이라면, 
#    Dockerfile 안에서 이름만 environment.yml 로 바꿔 처리해도 됩니다.
COPY environment.vml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# 3) 생성된 conda env 이름을 ENV 로 선언해서 자동 활성화
#    environment.yml 안에 `name: myenv` 로 돼 있다면 아래도 myenv 로.
ARG CONDA_ENV=myenv
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH

# 4) 작업 디렉터리 설정 및 소스 복사
WORKDIR /app
COPY . /app

# 5) (선택) 불필요한 캐시나 파일 무시
#    .dockerignore 에 __pycache__/, *.pyc, .git 등을 추가해 두시면 이미지가 가벼워집니다.

# 6) 외부에 노출할 포트
EXPOSE 8000

# 7) 컨테이너 시작 시 uvicorn 으로 FastAPI/ASGI 서버 실행
#    모듈 경로(main:app) 부분은 본인 프로젝트에 맞춰 바꿔주세요.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
