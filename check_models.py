import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print("--- 사용 가능한 Gemini 모델 목록 ---")
try:
    # 조건 없이 모델 전체를 가져와서 이름만 찍어봅니다.
    for m in genai.list_models():
        if 'gemini' in m.name:
            print(f"- 모델명: {m.name}")
            print(f"  (설명: {m.display_name})")
except Exception as e:
    print(f"목록 불러오기 실패: {e}")