"""OpenAI API 테스트 스크립트"""

from app.core.config import settings
from openai import OpenAI

def test_openai():
    print("="*50)
    print("OpenAI API 테스트")
    print("="*50)
    
    # API 키 확인
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    print(f"✅ API Key: {api_key[:20]}...")
    
    # OpenAI 클라이언트 생성
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI 클라이언트 생성 성공")
    except Exception as e:
        print(f"❌ 클라이언트 생성 실패: {e}")
        return
    
    # 간단한 API 호출 테스트
    try:
        print("\n🔄 API 호출 중...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
                {"role": "user", "content": "안녕하세요"}
            ],
            max_tokens=50
        )
        
        print("✅ API 호출 성공!")
        print(f"\n📊 사용 토큰:")
        print(f"  - 입력: {response.usage.prompt_tokens}")
        print(f"  - 출력: {response.usage.completion_tokens}")
        print(f"  - 총합: {response.usage.total_tokens}")
        print(f"\n💬 응답: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        if "quota" in str(e).lower():
            print("\n💡 무료 크레딧이 소진되었거나 결제 수단이 필요합니다.")
            print("   → https://platform.openai.com/account/billing")

if __name__ == "__main__":
    test_openai()
