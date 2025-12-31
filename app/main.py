"""FastAPI server for LangChain chatbot API."""

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    # EC2 배포 환경: app.* 경로 사용
    from app.config import settings  # type: ignore
    from app.core import (
        COLLECTION_NAME,
        CONNECTION_STRING,
        create_rag_chain,
        demo_mode,
        get_embeddings,
        get_llm,
        initialize_vector_store,
        interactive_mode,
        main as core_main,
        wait_for_postgres,
    )
    from app.router import chat_router
    from app.service.chat_service import ChatService, create_chat_service
    from app.service.midm_llm import build_midm_llm
except ImportError:
    # 로컬 개발 환경: backend.app.* 경로 사용
    from backend.app.config import settings  # type: ignore
    from backend.app.core import (
        COLLECTION_NAME,
        CONNECTION_STRING,
        create_rag_chain,
        demo_mode,
        get_embeddings,
        get_llm,
        initialize_vector_store,
        interactive_mode,
        main as core_main,
        wait_for_postgres,
    )
    from backend.app.router import chat_router
    from backend.app.service.chat_service import ChatService, create_chat_service
    from backend.app.service.midm_llm import build_midm_llm

app = FastAPI(title=settings.app_name, version=settings.app_version)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://www.elianayesol.com",
        "https://elianayesol.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
try:
    from app.router.chat_router import router as chat_api_router
except ImportError:
    from backend.app.router.chat_router import router as chat_api_router

app.include_router(chat_api_router)

# 전역 변수로 벡터 스토어와 RAG 체인 저장
vector_store = None
rag_chain = None


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment verification."""
    return {
        "status": "healthy",
        "service": "FastAPI LangChain RAG",
        "version": settings.app_version,
        "database": "connected" if CONNECTION_STRING else "not configured",
    }


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 LangChain 초기화."""
    global vector_store, rag_chain

    print("=" * 60)
    print("Initializing LangChain RAG System...")
    print("=" * 60)

    # Wait for PostgreSQL (Neon DB)
    print("\n[1/5] Checking PostgreSQL (Neon DB) connection...")
    wait_for_postgres()

    # Initialize embeddings (OpenAI 오류 시 FakeEmbeddings 사용)
    print("\n[2/5] Initializing embeddings...")
    try:
        # OpenAI API 키가 없거나 오류 발생 시 FakeEmbeddings 사용
        if not os.getenv("OPENAI_API_KEY"):
            print("No OpenAI API key, using FakeEmbeddings...")
            from langchain_core.embeddings import FakeEmbeddings

            embeddings = FakeEmbeddings(size=1536)
        else:
            # get_embeddings() 내부에서 이미 에러 핸들링됨
            embeddings = get_embeddings()
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Embeddings initialization failed: {e}")
        print("Using FakeEmbeddings instead...")
        from langchain_core.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(size=1536)

    # Initialize LLM (설정 기반: midm → openai → retrieval-only)
    print("\n[3/5] Initializing LLM...")
    chat_service_instance: Optional[ChatService] = None
    try:
        provider = settings.llm_provider.lower()
        midm_path = settings.local_model_dir or os.getenv("MIDM_PATH")

        if provider == "midm":
            print(f"Using local Mi:dm model (path={midm_path or 'default'})...")
            # ChatService 초기화 (PEFT QLoRA 사용)
            try:
                chat_service_instance = create_chat_service(
                    model_path=midm_path,
                    use_quantization=True,  # QLoRA: 4-bit 양자화 사용
                )
                # QLoRA 어댑터가 있으면 로드
                chat_service_instance.load_qlora_adapter()
                print("✓ ChatService initialized with QLoRA support")
                # LangChain용 LLM도 생성 (RAG chain용)
                llm = build_midm_llm(model_path=midm_path, max_new_tokens=256)
            except Exception as e:  # noqa: BLE001
                print(f"⚠ ChatService init failed: {e}")
                print("Falling back to basic Mi:dm LLM...")
                llm = build_midm_llm(model_path=midm_path, max_new_tokens=256)
        elif provider == "openai" and settings.openai_api_key:
            print("Using OpenAI ChatModel...")
            llm = get_llm()
        else:
            print("No LLM available (provider mismatch or missing key). Using retrieval only mode...")
            llm = None
    except Exception as e:  # noqa: BLE001
        print(f"⚠ LLM init failed: {e}")
        print("Using retrieval-only mode (no LLM)...")
        llm = None

    # Create vector store (에러 발생 시에도 서비스 시작 허용)
    print("\n[4/5] Creating vector store...")
    try:
        vector_store = initialize_vector_store(embeddings)
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Vector store initialization failed: {e}")
        print("⚠ Continuing without vector store (service will have limited functionality)")
        vector_store = None

    # Create RAG chain
    print("\n[5/5] Creating RAG chain...")
    if vector_store:
        rag_chain = create_rag_chain(vector_store, llm)
        if rag_chain:
            print("✓ RAG chain created!")
        else:
            print("⚠ RAG chain not available (retrieval-only mode)")
    else:
        print("⚠ RAG chain not available (no vector store)")
        rag_chain = None

    # 라우터에 의존성 주입
    try:
        from app.router.chat_router import set_dependencies
    except ImportError:
        from backend.app.router.chat_router import set_dependencies

    set_dependencies(vector_store, rag_chain, chat_service_instance)

    print("\n" + "=" * 60)
    print("API Server Ready!")
    print("=" * 60)


if __name__ == "__main__":
    # If running as script, check if we want CLI mode or API server
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run interactive/demo mode
        core_main()
    else:
        # Run API server
        import uvicorn

        # EC2 배포 시: app.main, 로컬 개발 시: backend.app.main
        import sys
        if "/home/ubuntu/app" in sys.path[0] or "/home/ubuntu/app" in str(sys.path):
            uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
        else:
            uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000)

