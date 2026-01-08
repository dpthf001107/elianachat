"""FastAPI 챗봇 라우터.

POST /api/chat
세션 ID, 메시지 리스트 등을 받아 대화형 응답 반환.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.service.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])

# 전역 변수로 벡터 스토어와 RAG 체인 참조 (main.py에서 초기화)
vector_store = None
rag_chain = None
chat_service: Optional[ChatService] = None


class ChatRequest(BaseModel):
    """채팅 요청 모델."""

    message: str


class ChatResponse(BaseModel):
    """채팅 응답 모델."""

    response: str


def set_dependencies(vs, chain, service: Optional[ChatService] = None):
    """main.py에서 초기화된 벡터 스토어와 RAG 체인, ChatService를 주입."""
    global vector_store, rag_chain, chat_service
    vector_store = vs
    rag_chain = chain
    chat_service = service


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇 API 엔드포인트."""
    global vector_store, rag_chain, chat_service

    try:
        query = request.message

        # ChatService가 있으면 우선 사용 (PEFT QLoRA 모델)
        if chat_service is not None:
            # 동기 함수를 비동기 컨텍스트에서 실행 (호환성 개선)
            import asyncio
            import sys

            # Python 3.9+에서는 asyncio.to_thread 사용, 그 이하는 run_in_executor 사용
            # max_new_tokens를 크게 설정하여 자유로운 응답 허용
            if sys.version_info >= (3, 9):
                response = await asyncio.to_thread(
                    chat_service.chat, query, 2048, 0.7, 0.9, None
                )
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: chat_service.chat(query, 2048, 0.7, 0.9, None)
                )
            return ChatResponse(response=response)

        # ChatService가 없으면 기존 RAG chain 사용
        if not vector_store:
            # 벡터 스토어가 없어도 기본 응답 반환 (서비스는 계속 동작)
            return ChatResponse(
                response="현재 벡터 스토어가 초기화되지 않았습니다. 시스템 관리자에게 문의해주세요."
            )

        if rag_chain:
            result = rag_chain.invoke({"input": query})
            answer = result.get("answer", result.get("result", "응답을 생성할 수 없습니다."))
            return ChatResponse(response=answer)
        else:
            # Fallback to simple retrieval
            results = vector_store.similarity_search(query, k=3)
            if results:
                response_text = "\n\n".join(
                    [f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)]
                )
                return ChatResponse(response=f"관련 문서를 찾았습니다:\n\n{response_text}")
            else:
                return ChatResponse(response="관련 문서를 찾을 수 없습니다.")

    except Exception as e:  # noqa: BLE001
        # 에러 상세 정보 로깅
        import traceback

        error_detail = str(e)
        error_traceback = traceback.format_exc()
        print(f"❌ Chat error: {error_detail}")
        print(f"Traceback:\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"오류가 발생했습니다: {error_detail}")


@router.get("/health")
async def health():
    """헬스 체크 엔드포인트."""
    return {"status": "ok", "vector_store_ready": vector_store is not None}
