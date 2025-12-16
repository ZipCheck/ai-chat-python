# ai-chat-python
FastAPI 기반 AI 챗봇 서버입니다. Spring Boot에서 스티커(리뷰) 데이터를 가져와 Gemini에 프롬프트로 전달해 답변을 생성합니다.

## Prerequisites
- Python 3.10+
- 가상환경 권장: venv
- 필수 환경 변수: `GOOGLE_API_KEY`, `SPRING_BOOT_API_BASE_URL`(기본값 `http://localhost:8080`)
- Git

## Setup
1) 클론: `git clone https://github.com/ZipCheck/ai-chat-python.git`
2) 가상환경 생성/활성화: `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3) 패키지 설치: `pip install -r requirements.txt`
4) `.env` 작성
   ```
   GOOGLE_API_KEY=your_gemini_key
   SPRING_BOOT_API_BASE_URL=http://localhost:8080
   ```
5) 실행: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

## Dependencies (requirements.txt)
- 의존성 설치: `pip install -r requirements.txt`
- 새 패키지 추가 후 버전 고정: `pip freeze > requirements.txt`로 갱신해 팀원이 같은 버전으로 맞출 수 있게 유지하세요.
- 아래 순서에 맞춰서 설치
```
brew install python@3.11          # 없다면 설치
python3.11 --version               # 3.11.x 확인
cd /Users/leejinhyung/ssafy/finalProject/ai-server
rm -rf .venv                       # 기존 3.9 venv 제거
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" "importlib-metadata>=6.0"
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API
- `GET /` : 헬스 체크
- `POST /api/deals/{deal_id}/chatbot`
  - Body 예시: `{"question": "매물 질문"}`
  - 동작: Spring `/api/stickers?dealId=...` 호출 → 리뷰로 프롬프트 구성 → Gemini 호출 → 답변 반환

## Git 워크플로 (최초 세팅)
```
git init        # .git 없을 때만
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/ZipCheck/ai-chat-python.git
git push -u origin main
```

## Notes
- `.env`, `venv/`, `__pycache__/` 등은 `.gitignore`에 포함하세요.
- Spring 엔드포인트나 응답 구조가 바뀌면 `main.py`에서 호출 URL/파싱 로직을 함께 수정해야 합니다.
