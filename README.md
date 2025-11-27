# LevelUpFit Pose Backend

AI 기반 운동 자세 분석 백엔드 서비스입니다.  
런지, 스쿼트 등 다양한 운동 동작의 비디오를 입력받아,  
MediaPipe 기반 포즈 추정 및 분석 결과(정확도, 가동범위, 피드백 등)를 제공합니다.

---

## 주요 기능

- **운동 자세 분석**: 업로드된 운동 영상을 프레임 단위로 분석하여,  
  무릎-발끝 관계, 가동범위, 수직 정렬, 수축/이완 속도 등 다양한 지표 산출
- **MediaPipe 기반 포즈 추정**: Mediapipe Pose 모델을 활용한 관절 좌표 추출
- **분석 결과 시각화**: 분석 결과가 반영된 영상(기준선, 랜드마크 등) 생성 및 저장
- **MinIO 연동**: 분석된 영상을 MinIO 오브젝트 스토리지에 저장 및 URL 제공
- **FastAPI 서버**: REST API로 영상 업로드 및 분석 결과 반환

---

## 폴더 구조

```
app/
  main.py                # FastAPI 엔트리포인트
  routers/pose.py        # API 라우터
  services/              # 분석 로직 및 유틸리티
    lunge_analyzer_ver2.py
    lunge_analyzer_level2.py
    lunge_analyzer_level3.py
    squat_analyzer.py
    mediapipe_service.py
    ...
  utils/                 # 각종 유틸리티
    angle_utils.py
    minio_client.py
    ...
  core/config.py         # 환경설정
```

---

## 설치 및 실행

1. **의존성 설치**
    ```bash
    pip install -r requirements.txt
    ```

2. **환경 변수 설정**
    - `.env` 파일에 MinIO 등 환경 변수 입력  
      (예시: `MINIO_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`)

3. **서버 실행**
    ```bash
    uvicorn app.main:app --reload
    ```

---

## API 사용 예시

- **POST /pose/analyze**
    - 운동 영상 파일 업로드 및 분석 요청
    - 파라미터: `exercise_id`, `level`, `feedback_id`, `file`
    - 응답: 분석 결과, 피드백, 분석 영상 URL 등

---

## 분석 로직 요약

- **포즈 추정**: MediaPipe로 관절 좌표 추출
- **정확도/가동범위 평가**: 무릎-발끝, 무릎-엉덩이 거리 등으로 평가
- **수축/이완 시간 산출**: 무릎-엉덩이 y좌표 변화 신호를 스무싱(Moving Average) 후,  
  신호의 방향 전환점(최대/최소)에서 구간을 나누고, 각 구간의 프레임 수를 FPS로 나눠 초 단위로 변환하여 평균 수축/이완 시간을 계산

---

## 참고

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MinIO](https://min.io/)

---

## 문의

- 이 저장소/서비스 관련 문의는 관리자에게 연락 바랍니다.
