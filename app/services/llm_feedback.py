# app/services/llm_feedback.py
"""
LLM(GPT)을 이용한 피드백 생성 모듈
런지 분석 결과를 바탕으로 자연스러운 피드백 텍스트를 생성합니다.
"""

from openai import OpenAI
from app.core.config import settings

# OpenAI 클라이언트 초기화
client = None

def get_openai_client():
    """OpenAI 클라이언트를 lazy initialization으로 가져옵니다."""
    global client
    if client is None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        client = OpenAI(api_key=api_key)
    return client

# 시스템 프롬프트 (lunge_prompt 내용)
SYSTEM_PROMPT = SYSTEM_PROMPT = """
당신은 사용자의 런지(lunge) 동작을 분석해서 피드백을 주는 전문 트레이너입니다.

규칙:
1. 항상 한국어 존댓말로 말합니다.
2. 피드백은 형식적인 문장 대신, 실제 트레이너가 말하듯 자연스럽고 따뜻하게 말합니다.
3. 아래 네 가지를 균형 있게 다룹니다.
   - 무릎-발끝 정렬 (knee_accuracy)
   - 수직 정렬 / 몸의 정렬 (vertical_score, 있는 경우)
   - 가동범위, 동작 깊이 (movement_range)
   - 수축/이완 속도 (movement_speed, 있는 경우)
4. 점수 해석 기준 (대략적인 가이드라인)
   - 90점 이상: 아주 좋음 → 구체적으로 무엇이 좋은지 칭찬
   - 60~89점: 보통, 개선 여지 있음 → 한두 가지 구체적인 팁 제시
   - 60점 미만: 많이 아쉬움 → 단점을 나열하지 말고, 핵심 1~2개만 짚어서 어떻게 바꾸면 좋아지는지 알려주기
5. 절대 숫자 점수만 나열하지 말고, 동작 모습이 어떤지 "그려지도록" 설명합니다.
6. 각 피드백은 최대 3개의 짧은 문단으로 작성하되,
   - 1문단: 전체적인 한 줄 요약 + 칭찬 포인트
   - 2문단: 무릎, 정렬, 가동범위, 속도 중 중요한 2~3가지를 골라 구체적인 코칭 포인트 제시
   - 3문단: 다음에 시도해볼 간단한 행동 지침 1~2개 + 응원 멘트
7. 같은 문장 패턴을 계속 반복하지 않도록, 문장 시작과 표현을 다양하게 바꿉니다.
8. 사용자에게 죄책감을 주지 말고, "이미 잘한 점 + 다음에 더 좋아질 수 있는 점"에 초점을 둡니다.
9. 점수나 시간 같은 숫자는 그대로 나열하지 말고,
   '거의 100점에 가까운', '2초보다 조금 빠른 편', '조금 빠른 리듬'처럼 말로 풀어서 설명합니다.
   소수점 둘째 자리까지의 숫자는 되도록 쓰지 않습니다.
10. 전체 길이는 3~6문장 안에서 끝내 주세요.
11. 가능하면 사용자가 몸으로 상상할 수 있는 리듬/느낌을 한 가지 이상 넣어 주세요.
    (예: '하나 둘 셋에 맞춰 내려가고 올라오는 느낌으로', '가슴을 살짝 펴고 시선은 정면을 본다는 느낌으로' 등)
12. 불릿 포인트나 번호 목록을 사용하지 말고, 자연스러운 문단 형식으로만 작성합니다.
13. 입력으로 주어지지 않은 항목(예: vertical_score, movement_speed 등)은 억지로 추측하거나 언급하지 않습니다.
14. knee_accuracy, vertical_score 같은 변수 이름은 그대로 말하지 말고,
    '무릎과 발끝 정렬 점수', '몸의 수직 정렬'처럼 자연스러운 한국어 표현으로만 설명합니다.
"""



def build_user_prompt(
    analysis_level: int,
    accuracy: float,
    movement_range: float,
    knee_accuracy: float = None,
    vertical_score: float = None,
    movement_speed: dict = None
) -> str:
    """
    분석 결과를 바탕으로 LLM에게 보낼 사용자 프롬프트를 생성합니다.
    
    Args:
        analysis_level: 분석 레벨 (1, 2, 3)
        accuracy: 전체 정확도 (0~100)
        movement_range: 가동범위 점수 (0~100)
        knee_accuracy: 무릎-발끝 정렬 정확도 (Level 2, 3에서 사용)
        vertical_score: 수직 정렬 점수 (Level 2, 3에서 사용)
        movement_speed: 수축/이완 속도 정보 dict (Level 3에서 사용)
    
    Returns:
        str: LLM에게 보낼 사용자 프롬프트
    """
    prompt = f"""아래는 런지 동작 분석 결과입니다.
이 데이터를 바탕으로 위에서 설명한 규칙대로 피드백 메시지를 작성해주세요.

- 분석 레벨: {analysis_level}

[공통 지표]
- accuracy: {accuracy:.1f} (0~100, 전체 정확도)
- movementRange: {movement_range:.1f} (0~100, 가동범위 점수)
"""

    # Level 2, 3에서 제공되는 세부 지표
    if analysis_level >= 2:
        knee_str = f"{knee_accuracy:.1f}" if knee_accuracy is not None else "제공되지 않음"
        vertical_str = f"{vertical_score:.1f}" if vertical_score is not None else "제공되지 않음"
        prompt += f"""
[레벨 {analysis_level} 세부 지표]
- knee_accuracy: {knee_str} (무릎-발끝 정렬 정확도)
- vertical_score: {vertical_str} (수직 정렬 점수, 어깨-엉덩이-반대쪽 무릎이 일직선일수록 높음)
"""

    # Level 3에서 제공되는 속도 정보
    if analysis_level == 3 and movement_speed is not None:
        prompt += f"""
[속도 정보]
- avgContractionTime: {movement_speed.get('avgContractionTime', 0):.2f}초 (평균 수축 시간, 이상적인 범위 2~3초)
- avgRelaxationTime: {movement_speed.get('avgRelaxationTime', 0):.2f}초 (평균 이완 시간, 이상적인 범위 2~3초)
- contractionPercent: {movement_speed.get('contractionPercent', 0)} (수축 속도 적절성, 0~100)
- relaxationPercent: {movement_speed.get('relaxationPercent', 0)} (이완 속도 적절성, 0~100)
"""

    prompt += """
출력 형식:
- 자연스러운 피드백 문단들로 작성해주세요.
- 불릿 포인트 대신, 문장/문단 형태로 작성해주세요.
- 길이는 3~6문장 정도로 해주세요.
"""

    return prompt


def generate_feedback(
    analysis_level: int,
    accuracy: float,
    movement_range: float,
    knee_accuracy: float = None,
    vertical_score: float = None,
    movement_speed: dict = None,
    fallback_func=None
) -> str:
    """
    LLM을 사용하여 피드백 텍스트를 생성합니다.
    
    Args:
        analysis_level: 분석 레벨 (1, 2, 3)
        accuracy: 전체 정확도 (0~100)
        movement_range: 가동범위 점수 (0~100)
        knee_accuracy: 무릎-발끝 정렬 정확도 (Level 2, 3에서 사용)
        vertical_score: 수직 정렬 점수 (Level 2, 3에서 사용)
        movement_speed: 수축/이완 속도 정보 dict (Level 3에서 사용)
        fallback_func: LLM 호출 실패시 사용할 기존 피드백 함수 (callable)
    
    Returns:
        str: 생성된 피드백 텍스트
    """
    try:
        openai_client = get_openai_client()
        
        user_prompt = build_user_prompt(
            analysis_level=analysis_level,
            accuracy=accuracy,
            movement_range=movement_range,
            knee_accuracy=knee_accuracy,
            vertical_score=vertical_score,
            movement_speed=movement_speed
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # 비용 효율적인 모델 사용
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        feedback_text = response.choices[0].message.content.strip()
        print(f"[LLM] 피드백 생성 성공 (Level {analysis_level})")
        return feedback_text
        
    except Exception as e:
        print(f"[LLM] 피드백 생성 실패: {e}")
        
        # fallback 함수가 제공되면 기존 방식으로 피드백 생성
        if fallback_func is not None:
            print("[LLM] 기존 피드백 함수로 fallback")
            return fallback_func()
        
        # fallback 함수가 없으면 기본 메시지 반환
        return "피드백을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요."


# Level별 편의 함수들
def generate_feedback_level1(accuracy: float, movement_range: float) -> str:
    """Level 1 피드백 생성"""
    def fallback():
        feedback = []
        if accuracy >= 90:
            feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
        else:
            feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
        if movement_range >= 80:
            feedback.append("가동범위가 충분합니다.")
        else:
            feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
        return "\n".join(feedback)
    
    return generate_feedback(
        analysis_level=1,
        accuracy=accuracy,
        movement_range=movement_range,
        fallback_func=fallback
    )


def generate_feedback_level2(
    accuracy: float,
    movement_range: float,
    knee_accuracy: float,
    vertical_score: float
) -> str:
    """Level 2 피드백 생성"""
    def fallback():
        feedback = []
        if knee_accuracy >= 90:
            feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
        else:
            feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
        if vertical_score >= 90:
            feedback.append("어깨, 엉덩이, 반대쪽 발이 잘 수직을 이루고 있습니다.")
        elif vertical_score >= 60:
            feedback.append("수직 정렬이 약간 부족합니다.\n엉덩이와 어깨, 반대쪽 발이 일직선이 되도록 신경써보세요.")
        else:
            feedback.append("수직 정렬이 많이 부족합니다.\n자세를 더 곧게 유지하세요.")
        if movement_range >= 80:
            feedback.append("가동범위가 충분합니다.")
        else:
            feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
        return "\n".join(feedback)
    
    return generate_feedback(
        analysis_level=2,
        accuracy=accuracy,
        movement_range=movement_range,
        knee_accuracy=knee_accuracy,
        vertical_score=vertical_score,
        fallback_func=fallback
    )


def generate_feedback_level3(
    accuracy: float,
    movement_range: float,
    knee_accuracy: float,
    vertical_score: float,
    movement_speed: dict
) -> str:
    """Level 3 피드백 생성"""
    def fallback():
        feedback = []
        if knee_accuracy >= 90:
            feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
        else:
            feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
        if vertical_score >= 90:
            feedback.append("수직 정렬이 매우 좋습니다.")
        elif vertical_score >= 60:
            feedback.append("수직 정렬이 약간 부족합니다.")
        else:
            feedback.append("수직 정렬이 많이 부족합니다.")
        if movement_range >= 80:
            feedback.append("가동범위가 충분합니다.")
        else:
            feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
        if movement_speed["contractionPercent"] >= 80 and movement_speed["relaxationPercent"] >= 80:
            feedback.append("수축과 이완 속도가 적절합니다.")
        else:
            if movement_speed["contractionPercent"] < 80:
                feedback.append(f"수축 속도가 적절하지 않습니다. 평균 수축 시간: {movement_speed['avgContractionTime']:.2f}초")
            if movement_speed["relaxationPercent"] < 80:
                feedback.append(f"이완 속도가 적절하지 않습니다. 평균 이완 시간: {movement_speed['avgRelaxationTime']:.2f}초")
        return "\n".join(feedback)
    
    return generate_feedback(
        analysis_level=3,
        accuracy=accuracy,
        movement_range=movement_range,
        knee_accuracy=knee_accuracy,
        vertical_score=vertical_score,
        movement_speed=movement_speed,
        fallback_func=fallback
    )
