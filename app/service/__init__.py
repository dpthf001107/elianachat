"""Backend application package exposing core utilities."""

try:
    from app.core import (  # EC2
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
    from backend.app.core import (  # Local
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

