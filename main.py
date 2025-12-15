from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

# 환경 변수 가져오기 (예: Spring 서버 URL, Gemini API Key)
SPRING_BOOT_API_BASE_URL = os.getenv("SPRING_BOOT_API_BASE_URL", "http://localhost:8080")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Google API Key for Gemini

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

class ChatbotRequest(BaseModel):
    question: str

class ChatbotResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Python AI Server is running!"}

@app.post("/api/deals/{deal_id}/chatbot", response_model=ChatbotResponse)
async def get_chatbot_answer(deal_id: int, request: ChatbotRequest):
    # [Step 1] Spring Boot에서 스티커(리뷰) 데이터 가져오기 (Retrieval)
    spring_api_url = f"{SPRING_BOOT_API_BASE_URL}/api/stickers"
    
    try:
        # Spring 서버에 GET 요청 보냄 (예: http://localhost:8080/api/stickers?dealId=1)
        response = requests.get(spring_api_url, params={"dealId": deal_id})
        response.raise_for_status() # 200 OK가 아니면 에러 발생
        
        # Spring에서 준 JSON 데이터 파싱 (Spring 응답 구조에 따라 수정 필요)
        # 예: {"data": [{"description": "조용해요"}, ...]} 라고 가정
        api_result = response.json()
        stickers_data = api_result.get("data", []) 
        
    except requests.exceptions.RequestException as e:
        print(f"Spring Boot 통신 오류: {e}")
        return {"answer": "죄송해요, 매물 정보를 가져오는 중에 문제가 생겼어요."}

    # 데이터가 아예 없는 경우 처리
    if not stickers_data:
        return {"answer": "아직 이 매물에 대한 주민들의 상세한 정보가 없어요."}
    
    # [Step 2] 스티커 설명들을 하나의 긴 문자열로 합치기 (Context 구성)
    # 리스트에 있는 description만 뽑아서 줄바꿈(\n)으로 연결
    context = "\n".join([s.get("description", "") for s in stickers_data if s.get("description")])

    if not context.strip():
        return {"answer": "아직 이 매물에 대한 유효한 주민 리뷰 내용이 없어요."}

    # [Step 3] LLM 프롬프트 만들기 (Prompt Engineering)
    # AI에게 역할(Persona)과 데이터(Context)를 줌
    prompt = f"""
    너는 특정 매물에 대한 실제 주민 리뷰를 바탕으로 질문에 답변해주는 친절한 AI 부동산 요정이야. 
    아래 [리뷰 내용]을 참고해서 사용자의 [질문]에 답해줘. 
    
    만약 리뷰 내용으로 알 수 없는 질문이라면, 지어내지 말고 
    "죄송하지만 해당 정보는 리뷰에 없어서 답변해드리기 어려워요." 라고 솔직하게 말해줘.

    [리뷰 내용]
    {context}

    [질문]
    {request.question}

    [답변]
    """

    # [Step 4] Gemini에게 질문하고 답 받기 (Generation)
    try:
        ai_response = model.generate_content(prompt)
        return {"answer": ai_response.text}
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 답변 생성 중 오류가 발생했습니다.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)