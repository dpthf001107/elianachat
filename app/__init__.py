"""Backend application package exposing core utilities."""

try:
    # EC2 환경
    from app.core import (
        COLLECTION_NAME,
        CONNECTION_STRING,
        create_rag_chain,
        demo_mode,
        get_embeddings,
        get_llm,
        initialize_vector_store,
        interactive_mode,
        main,
        wait_for_postgres,
    )
except ImportError:
    # 로컬 환경
    from backend.app.core import (
        COLLECTION_NAME,
        CONNECTION_STRING,
        create_rag_chain,
        demo_mode,
        get_embeddings,
        get_llm,
        initialize_vector_store,
        interactive_mode,
        main,
        wait_for_postgres,
    )

__all__ = [
    "COLLECTION_NAME",
    "CONNECTION_STRING",
    "create_rag_chain",
    "demo_mode",
    "get_embeddings",
    "get_llm",
    "initialize_vector_store",
    "interactive_mode",
    "main",
    "wait_for_postgres",
]

