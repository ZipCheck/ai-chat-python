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
    apt_id: str

class StickerData(BaseModel):
    stickerId: int
    description: str

class IndexDataRequest(BaseModel):
    apt_id: str
    stickers: list[StickerData]

class ChatbotRequest(BaseModel):
    question: str

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
    apt_id = request.apt_id
    #deal_id = request.deal_id
    stickers_data = request.stickers

    if not stickers_data:
        return {"message": f"Deal ID {apt_id}에 대한 유효한 리뷰가 없어 인덱싱할 내용이 없습니다."}

    # 인덱싱할 문서, 메타데이터, ID 리스트 준비
    documents = []
    metadatas = []
    ids = []

    for sticker in stickers_data:
        documents.append(sticker.description)
        metadatas.append({"apt_id": apt_id})
        ids.append(f"sticker_{sticker.stickerId}") # 각 문서를 고유하게 식별할 ID

    # ChromaDB에 데이터 추가/업데이트 (Upsert)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    return {"message": f"Deal ID {apt_id}에 대한 {len(documents)}개의 리뷰가 성공적으로 인덱싱되었습니다."}


@app.post("/api/apartments/{apt_id}/chatbot", response_model=ChatbotResponse)
async def get_chatbot_answer(apt_id: str, request: ChatbotRequest):
    """
    사용자 질문을 기반으로 Vector DB에서 관련 리뷰를 검색하고,
    이를 바탕으로 Gemini 모델이 답변을 생성합니다.
    """
    # [Step 1] 사용자 질문을 벡터로 변환 (Embedding)
    question_embedding = genai.embed_content(
        model="models/embedding-001",
        content=request.question,
        task_type="RETRIEVAL_QUERY"
    )[ "embedding"]

    # [Step 2] Vector DB에서 관련 문서 검색 (Retrieval)
    # 해당 deal_id를 가진 리뷰 중에서 질문과 유사한 3개 문서를 찾음
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        where={"apt_id": apt_id} # 필터링 조건
    )
    
    retrieved_documents = results['documents'][0]

    # 검색된 리뷰가 없는 경우
    if not retrieved_documents:
        return {"answer": "아직 이 매물에 대한 주민들의 상세한 정보가 없거나, 질문과 관련된 리뷰를 찾지 못했어요.", "score": 0}

    # [Step 3] 검색된 리뷰를 바탕으로 Context 구성 (Augmentation)
    context = "\n".join(retrieved_documents)

    # [Step 4] LLM 프롬프트 구성 (Prompt Engineering)
    prompt = f"""
    너는 특정 매물에 대한 실제 주민 리뷰를 바탕으로 질문에 답변해주는 친절한 AI 부동산 요정이야. 
    아래 [리뷰 내용]을 참고해서 사용자의 [질문]에 답해줘. 리뷰 내용은 주민들이 직접 작성한 후기야.

    **응답 형식:**
    결과는 반드시 다음 JSON 형식으로 출력해줘:
    {{
        "answer": "답변 내용 (좋았던점과 나쁜점을 구분하여 포함)",
        "score": 85 (0~100점 사이 정수)
    }}
    반드시 순수한 JSON 문자열만 출력해. 마크다운 코드 블록(```json)이나 기타 설명은 절대 포함하지 마.

    [지시사항]
    1. 'answer' 필드에 답변할때 **좋았던점**과 **나쁜점**을 개행으로 구분해서 작성해줘.
    2. **답변은 최대한 구체적이고 상세하게 작성해줘. 단순히 한두 줄로 끝내지 말고, 리뷰 내용을 충분히 인용해서 각 항목(좋았던점, 나쁜점)당 최소 3문장 이상으로 풍부하게 설명해줘.**
    3. 'score' 필드에는 좋은점과 나쁜점의 비율을 계산해서 0~100점 사이의 정수 점수를 산정해줘.
    
    만약 리뷰 내용으로 알 수 없는 질문이라면, 절대로 내용을 지어내지 말고 
    "죄송하지만 해당 정보는 리뷰에 없어서 답변해드리기 어려워요." 라고 answer 필드에 솔직하게 말하고, score 필드는 0으로 줘.

    [리뷰 내용]
    {context}

    [질문]
    {request.question}
    """

    # [Step 5] Gemini 모델에게 질문하고 답변 받기 (Generation)
    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        ai_response = generation_model.generate_content(prompt, generation_config=generation_config)
        
        # JSON 응답 파싱
        response_data = json.loads(ai_response.text)
        answer = response_data.get("answer", "AI 응답 파싱 오류.")
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