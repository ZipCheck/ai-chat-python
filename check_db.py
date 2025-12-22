import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB 경로 설정 (main.py와 동일하게)
client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection(name="real-estate-reviews")
    
    # 모든 데이터 조회 (최대 10개)
    results = collection.get()
    
    print(f"총 데이터 개수: {len(results['ids'])}")
    
    if len(results['ids']) == 0:
        print("데이터가 하나도 없습니다. 인덱싱 API를 먼저 호출해주세요.")
    else:
        print("\n--- 저장된 데이터 샘플 (최대 5개) ---")
        for i in range(min(5, len(results['ids']))):
            print(f"ID: {results['ids'][i]}")
            print(f"Metadata: {results['metadatas'][i]}")
            print(f"Document: {results['documents'][i]}")
            print("-" * 30)

except Exception as e:
    print(f"컬렉션을 불러오는 중 오류 발생: {e}")
    print("아직 컬렉션이 생성되지 않았거나 경로가 잘못되었을 수 있습니다.")
