import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.generativeai as genai

# --- 초기 설정 ---

# .env 파일에서 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# 환경 변수 가져오기
SPRING_BOOT_API_BASE_URL = os.getenv("SPRING_BOOT_API_BASE_URL", "http://localhost:8080")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Google Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

# --- AI 모델 설정 ---

# 사용할 Gemini 모델 설정
generation_model = genai.GenerativeModel('gemini-2.5-flash')

# --- Pydantic 모델 정의 ---

class StickerData(BaseModel):
    stickerId: int
    description: str

class ChatbotRequest(BaseModel):
    apt_seq: str | None = None
    question: str
    stickers: list[StickerData] | None = None

class ChatbotResponse(BaseModel):
    answer: str
    score: int

# --- API 엔드포인트 ---

@app.get("/")
async def root():
    return {"message": "Python AI Server is running!"}


@app.post("/api/apartments/{apt_id}/chatbot", response_model=ChatbotResponse)
async def get_chatbot_answer(apt_id: str, request: ChatbotRequest):
    """
    사용자 질문을 기반으로 Gemini 모델이 답변을 생성합니다.
    """
    stickers_count = len(request.stickers) if request.stickers else 0
    print(
        "[chatbot] "
        f"apt_id={apt_id} apt_seq={request.apt_seq} "
        f"question={request.question} stickers_count={stickers_count}"
    )
    if request.stickers:
        print(f"[chatbot] stickers={request.stickers}")

    stickers_text = "\n".join(
        f"- {sticker.description}" for sticker in (request.stickers or [])
    )

    # [Step 1] LLM 프롬프트 구성 (Prompt Engineering)
    prompt = f"""
        너는 부동산 관련 질문에 답변해주는 친절한 AI야.

        아래에 제공되는 [리뷰/스티커 내용]만을 근거로 답변해야 해.
        다른 일반 지식이나 추측은 절대 사용하지 마.
        매물 ID는 참고용일 뿐이며, 매물의 상세 정보는 제공되지 않았어.

        만약 질문에 답하기에 충분한 정보가 없다면,
        절대로 내용을 지어내지 말고 아래 문장을 그대로 사용해:
        "죄송하지만 해당 정보는 제공되지 않아서 답변해드리기 어려워요."

        ---

        ### [출력 형식 – 매우 중요]

        반드시 **아래 JSON 형식 그대로** 출력해.
        - JSON 외의 텍스트, 설명, 마크다운(```), 코드블록을 **절대 포함하지 마**
        - JSON은 **반드시 한 줄**로 출력
        - 문자열 안에서 줄바꿈은 `\\n` 으로 표현
        - 큰따옴표가 필요하면 `\\\"` 로 이스케이프

        출력 예시는 형식 참고용이며, 그대로 복사하지 마:
        {{
            "answer":"좋았던점: ... \\n나쁜점: ...",
            "score":85
        }}

        ---

        ### [answer 작성 규칙]

        1. answer에는 반드시 아래 두 항목을 포함해야 해:
        - **좋았던점**
        - **나쁜점**
        2. 두 항목은 반드시 `\\n`(개행)으로 구분해서 작성해.
        3. 리뷰/스티커 내용이 충분하다면:
        - 너무 구체적이지도, 너무 짧지도 않게 **중간 수준으로 요약**
        4. 리뷰/스티커 내용이 매우 적다면:
        - 각 항목을 **1~2문장 정도로 간단히 요약**

        ---

        ### [score 산정 규칙]

        - score는 0~100 사이의 정수여야 해
        - 리뷰/스티커에서 나타난 **긍정 요소와 부정 요소의 비율**을 기준으로 점수를 계산해
        - 긍정이 많을수록 높은 점수, 부정이 많을수록 낮은 점수
        - 정보 부족으로 답변할 수 없는 경우:
        - answer는 지정된 사과 문구 사용
        - score는 반드시 0


    [리뷰/스티커 내용]
    {stickers_text}

    [매물 ID]
    {apt_id}

    [질문]
    {request.question}
    """
    

    # [Step 2] Gemini 모델에게 질문하고 답변 받기 (Generation)
    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        ai_response = generation_model.generate_content(prompt, generation_config=generation_config)
        
        # JSON 응답 파싱
        response_data = json.loads(ai_response.text)
        answer = response_data.get("answer", "AI 응답 파싱 오류.")
        answer = answer.replace("\\n", "\n")
        if "좋았던점" in answer and "나쁜점" in answer and "\n" not in answer:
            answer = answer.replace("나쁜점", "\n나쁜점", 1)
        score = response_data.get("score", 0) # 파싱 오류 시 기본 점수 0

        return {"answer": answer, "score": score}
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 응답을 파싱하는 중 오류가 발생했습니다.")
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 답변 생성 중 오류가 발생했습니다.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
