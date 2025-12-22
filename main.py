import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

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

# --- AI 모델 및 Vector DB 설정 ---

# 사용할 Gemini 모델 설정
generation_model = genai.GenerativeModel('gemini-2.5-flash')
embedding_model = genai.GenerativeModel('models/embedding-001')
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)

# ChromaDB 클라이언트 설정 (메모리 기반)
# 서버가 재시작되면 데이터는 사라집니다. 영구 저장을 위해서는 경로를 지정해야 합니다.
# client = chromadb.PersistentClient(path="/path/to/db")
client = chromadb.PersistentClient(path="./chroma_db")
# client = chromadb.Client()

# Vector DB 컬렉션 생성 (없으면 생성)
# 유사도 검색을 위해 cosine 거리를 사용하도록 설정
collection = client.get_or_create_collection(
    name="real-estate-reviews",
    metadata={"hnsw:space": "cosine"},
    embedding_function=gemini_ef
)

# --- Pydantic 모델 정의 ---

class IndexRequest(BaseModel):
    apt_seq: str

class StickerData(BaseModel):
    stickerId: int
    description: str

class IndexDataRequest(BaseModel):
    apt_seq: str
    stickers: list[StickerData]

class ChatbotRequest(BaseModel):
    question: str

class ReportRequest(BaseModel):
    question: str
    stickers: list[StickerData]

class ChatbotResponse(BaseModel):
    answer: str
    score: int

# --- API 엔드포인트 ---

@app.get("/")
async def root():
    return {"message": "Python AI Server with RAG and Vector DB is running!"}

@app.post("/api/index-reviews-with-data", status_code=201)
async def index_reviews_with_data(request: IndexDataRequest):
    """
    Spring 서버로부터 직접 리뷰 데이터를 받아 Vector DB에 저장(인덱싱)합니다.
    """
    apt_seq = request.apt_seq
    stickers_data = request.stickers

    if not stickers_data:
        return {"message": f"Apt Seq {apt_seq}에 대한 유효한 리뷰가 없어 인덱싱할 내용이 없습니다."}

    # 인덱싱할 문서, 메타데이터, ID 리스트 준비
    documents = []
    metadatas = []
    ids = []

    for sticker in stickers_data:
        documents.append(sticker.description)
        metadatas.append({"apt_seq": apt_seq})
        ids.append(f"sticker_{sticker.stickerId}") # 각 문서를 고유하게 식별할 ID

    # ChromaDB에 데이터 추가/업데이트 (Upsert)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    return {"message": f"Apt Seq {apt_seq}에 대한 {len(documents)}개의 리뷰가 성공적으로 인덱싱되었습니다."}


@app.post("/api/apartments/{apt_seq}/chatbot", response_model=ChatbotResponse)
async def generate_apt_report(apt_seq: str, request: ReportRequest):
    """
    Spring 서버로부터 받은 aptSeq, 질문, 스티커 데이터를 기반으로
    아파트 단지에 대한 종합적인 분석 리포트를 생성합니다.
    """
    stickers_data = request.stickers
    question = request.question

    # 스티커 데이터가 없는 경우 처리
    if not stickers_data:
        return {"answer": "분석할 스티커 리뷰 데이터가 없습니다.", "score": 0}

    # [Step 1] Context 구성 (Sticker 데이터를 텍스트로 변환)
    context = "\n".join([sticker.description for sticker in stickers_data])

    print(f"리포트 요청 apt_seq: {apt_seq}") # DEBUG

    # [Step 2] LLM 프롬프트 구성
    prompt = f"""
    너는 아파트 단지 거주 환경을 분석해주는 AI 전문가야.
    아래 [주민들의 스티커 리뷰]를 종합적으로 분석해서, 사용자의 [질문]에 맞춰 보고서를 작성해줘.

    **응답 형식:**
    결과는 반드시 다음 JSON 형식으로 출력해줘:
    {{
        "answer": "보고서 내용 (요약 및 분석)",
        "score": 85 (0~100점 사이 정수, 거주 만족도 점수)
    }}
    반드시 순수한 JSON 문자열만 출력해. 마크다운 코드 블록(```json)이나 기타 설명은 절대 포함하지 마.

    [지시사항]
    1. '{apt_seq} 아파트 단지에 대한 주민들의 스티커 리뷰들을 종합적으로 요약하고, 전반적인 거주 환경에 대한 보고서를 작성해줘.' 라는 목표에 맞춰 답변해줘.
    2. 주민들의 리뷰 내용을 바탕으로 **장점**과 **단점**을 명확하게 파악하고, 이를 종합하여 서술해줘.
    3. **단순 나열이 아닌, '교통이 편리하지만 소음이 있다'와 같이 유기적인 문장으로 작성해줘.**
    4. 'score' 필드에는 리뷰의 긍정/부정 비율을 고려하여 0~100점 사이의 거주 만족도 점수를 산정해줘.
    
    [주민들의 스티커 리뷰]
    {context}

    [질문]
    {question}
    """

    # [Step 3] Gemini 모델에게 요청 (Generation)
    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        ai_response = generation_model.generate_content(prompt, generation_config=generation_config)
        
        # JSON 응답 파싱
        response_data = json.loads(ai_response.text)
        answer = response_data.get("answer", "AI 응답 파싱 오류.")
        score = response_data.get("score", 0)

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