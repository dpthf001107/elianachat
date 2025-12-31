"""LangChain RAG application with PGVector integration."""

import os
import time
from typing import List, Optional

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

try:
    from app.config import settings  # type: ignore  # EC2
except ImportError:
    from backend.app.config import settings  # type: ignore  # Local

# Try to import OpenAI, fallback to fake if not available
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Database connection string from .env
CONNECTION_STRING = settings.database_url or ""

# Debug: Print DATABASE_URL status
print(f"[DEBUG] settings.database_url = {settings.database_url}")
print(f"[DEBUG] CONNECTION_STRING = {CONNECTION_STRING}")
print(f"[DEBUG] os.getenv('DATABASE_URL') = {os.getenv('DATABASE_URL')}")

# Collection name for vector store
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain_collection")

# OpenAI API key (from settings, which reads from .env)
OPENAI_API_KEY = settings.openai_api_key or ""


def wait_for_postgres(max_retries: int = 30, delay: int = 2) -> None:
    """Wait for PostgreSQL (Neon DB) to be ready."""
    if not CONNECTION_STRING:
        print("⚠ DATABASE_URL not set, skipping PostgreSQL connection check")
        return

    import psycopg2

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(CONNECTION_STRING)
            conn.close()
            print("✓ PostgreSQL (Neon DB) is ready!")
            return
        except Exception as e:  # noqa: PERF203
            if i < max_retries - 1:
                print(f"Waiting for PostgreSQL (Neon DB)... ({i+1}/{max_retries})")
                time.sleep(delay)
            else:
                raise ConnectionError(f"Failed to connect to PostgreSQL (Neon DB): {e}")  # noqa: EM101


def get_embeddings():
    """Get embeddings model."""
    if HAS_OPENAI and OPENAI_API_KEY:
        print("Using OpenAI embeddings...")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # type: ignore[arg-type]
    else:
        print("Using FakeEmbeddings (no API key required)...")
        return FakeEmbeddings(size=1536)


def get_llm() -> Optional[BaseLanguageModel]:
    """Get LLM model."""
    if HAS_OPENAI and OPENAI_API_KEY:
        print("Using OpenAI ChatModel...")
        # api_key는 SecretStr 타입을 요구하지만, str도 런타임에 작동함
        # 프롬프트 템플릿에서 한글 응답을 강제하므로 여기서는 기본 설정만 사용
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,  # 더 자연스러운 응답을 위해 temperature 증가
            api_key=OPENAI_API_KEY,  # type: ignore[arg-type]
        )
    else:
        print("No LLM available (OpenAI API key not set). Using retrieval only mode...")
        return None


def initialize_vector_store(embeddings) -> PGVector:
    """Initialize PGVector store with sample documents."""
    if not CONNECTION_STRING:
        raise ValueError("DATABASE_URL is required for PGVector. Please set it in .env file.")

    documents = [
        Document(
            page_content="LangChain은 LLM 기반 애플리케이션을 쉽게 만들 수 있도록 도와주는 프레임워크로, 체인과 도구, 에이전트 등을 추상화해 제공합니다.",
            metadata={"source": "langchain_intro", "topic": "framework"},
        ),
        Document(
            page_content="pgvector는 PostgreSQL에서 벡터 유사도 검색을 지원하는 확장으로, 임베딩을 저장하고 효율적으로 검색할 수 있게 해줍니다.",
            metadata={"source": "pgvector_docs", "topic": "database"},
        ),
        Document(
            page_content="RAG(Retrieval-Augmented Generation)는 검색된 문서를 활용해 더 정확한 생성 결과를 만드는 방식입니다.",
            metadata={"source": "rag_concept", "topic": "ai"},
        ),
        Document(
            page_content="벡터 데이터베이스는 텍스트나 이미지의 임베딩을 저장하고, 코사인 유사도 같은 거리 기반으로 의미 검색을 수행합니다.",
            metadata={"source": "vector_db", "topic": "database"},
        ),
        Document(
            page_content="임베딩은 텍스트의 의미를 수치로 표현한 것으로, 유사한 의미의 문장은 서로 가까운 벡터 값을 갖습니다.",
            metadata={"source": "embeddings", "topic": "ai"},
        ),
    ]

    print(f"Creating PGVector store with {len(documents)} documents...")
    vector_store = PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print("✓ PGVector store created!")
    return vector_store


def create_rag_chain(
    vector_store: PGVector, llm: Optional[BaseLanguageModel]
) -> Optional[Runnable]:
    """Create RAG chain using LCEL."""
    if llm is None:
        return None

    try:
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain

        prompt_template = """당신은 한국어로 답변하는 AI 어시스턴트입니다. 모든 답변은 반드시 한글로 작성해주세요.

다음 컨텍스트를 사용하여 질문에 답변해주세요.
제공된 컨텍스트를 바탕으로 상세하고 포괄적인 답변을 제공해주세요.
컨텍스트에 질문에 대한 충분한 정보가 없는 경우, 컨텍스트에 충분한 정보가 없다고 말할 수 있지만, 여전히 알고 있는 내용을 바탕으로 도움이 되는 답변을 제공해주세요.

중요: 반드시 한글로만 답변해주세요. 영어로 답변하지 마세요.

컨텍스트:
{context}

질문: {input}

답변 (한글로 작성):"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        chain = create_retrieval_chain(retriever, document_chain)
        return chain
    except ImportError:
        # Fallback to deprecated API if new one is not available
        from langchain_classic.chains.retrieval_qa.base import RetrievalQA

        prompt_template = """당신은 한국어로 답변하는 AI 어시스턴트입니다. 모든 답변은 반드시 한글로 작성해주세요.

다음 컨텍스트를 사용하여 질문에 답변해주세요.
제공된 컨텍스트를 바탕으로 상세하고 포괄적인 답변을 제공해주세요.
컨텍스트에 질문에 대한 충분한 정보가 없는 경우, 컨텍스트에 충분한 정보가 없다고 말할 수 있지만, 여전히 알고 있는 내용을 바탕으로 도움이 되는 답변을 제공해주세요.

중요: 반드시 한글로만 답변해주세요. 영어로 답변하지 마세요.

컨텍스트:
{context}

질문: {question}

답변 (한글로 작성):"""

        prompt = PromptTemplate.from_template(prompt_template)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return chain


def interactive_mode(vector_store: PGVector, rag_chain: Optional[Runnable]) -> None:
    """Run interactive Q&A mode."""
    print("\n" + "=" * 60)
    print("Interactive Q&A Mode")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            query = input("\n💬 Your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not query:
                continue

            # If RAG chain is available, use it
            if rag_chain:
                print("\n🤔 Thinking...")
                try:
                    result = rag_chain.invoke({"input": query})
                    answer = result.get("answer", result.get("result", "No answer generated"))
                    print(f"\n💡 Answer:\n{answer}")
                    print("\n📚 Sources:")
                    context = result.get("context", [])
                    if isinstance(context, list):
                        for i, doc in enumerate(context, 1):
                            if hasattr(doc, "page_content"):
                                print(f"  {i}. {doc.page_content[:100]}...")
                                print(f"     Metadata: {doc.metadata}")
                    source_docs = result.get("source_documents", [])
                    if source_docs:
                        for i, doc in enumerate(source_docs, 1):
                            print(f"  {i}. {doc.page_content[:100]}...")
                            print(f"     Metadata: {doc.metadata}")
                except Exception as e:  # noqa: BLE001
                    print(f"\n❌ Error in RAG chain: {e}")
                    # Fallback to simple retrieval
                    fallback_results: List[Document] = vector_store.similarity_search(query, k=3)
                    print(f"\n📄 Found {len(fallback_results)} relevant documents:")
                    for i, doc in enumerate(fallback_results, 1):
                        print(f"\n  {i}. {doc.page_content}")
                        print(f"     Metadata: {doc.metadata}")
            else:
                # Fallback to simple retrieval
                print("\n🔍 Searching...")
                results: List[Document] = vector_store.similarity_search(query, k=3)
                print(f"\n📄 Found {len(results)} relevant documents:")
                for i, doc in enumerate(results, 1):
                    print(f"\n  {i}. {doc.page_content}")
                    print(f"     Metadata: {doc.metadata}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Error: {e}")


def demo_mode(vector_store: PGVector, rag_chain: Optional[Runnable]) -> None:
    """Run demo mode with sample queries."""
    print("\n" + "=" * 60)
    print("Hello World Demo")
    print("=" * 60)

    # Sample queries
    sample_queries = [
        "What is LangChain?",
        "What is pgvector?",
        "What is RAG?",
    ]

    for query in sample_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        # If RAG chain is available, use it
        if rag_chain:
            print("\n🤔 Thinking...")
            try:
                result = rag_chain.invoke({"input": query})
                answer = result.get("answer", result.get("result", "No answer generated"))
                print(f"\n💡 Answer:\n{answer}")
                print("\n📚 Sources:")
                context = result.get("context", [])
                if isinstance(context, list):
                    for i, doc in enumerate(context, 1):
                        if hasattr(doc, "page_content"):
                            print(f"  {i}. {doc.page_content[:100]}...")
                source_docs = result.get("source_documents", [])
                if source_docs:
                    for i, doc in enumerate(source_docs, 1):
                        print(f"  {i}. {doc.page_content[:100]}...")
            except Exception as e:  # noqa: BLE001
                print(f"\n❌ Error in RAG chain: {e}")
                # Fallback to simple retrieval
            fallback_results: List[Document] = vector_store.similarity_search(query, k=2)
            print(f"\n📄 Found {len(fallback_results)} relevant documents:")
            for i, doc in enumerate(fallback_results, 1):
                    print(f"\n  {i}. {doc.page_content}")
        else:
            # Fallback to simple retrieval
            print("\n🔍 Searching...")
            results: List[Document] = vector_store.similarity_search(query, k=2)
            print(f"\n📄 Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n  {i}. {doc.page_content}")

    print("\n" + "=" * 60)
    print("Hello World Demo completed! 🎉")
    print("=" * 60)


def main() -> None:
    """Main function to run the LangChain RAG application."""
    print("=" * 60)
    print("LangChain RAG System with PGVector")
    print("=" * 60)

    # Wait for PostgreSQL (Neon DB)
    print("\n[1/6] Checking PostgreSQL (Neon DB) connection...")
    wait_for_postgres()

    # Initialize embeddings
    print("\n[2/6] Initializing embeddings...")
    embeddings = get_embeddings()

    # Initialize LLM
    print("\n[3/6] Initializing LLM...")
    llm = get_llm()

    # Create vector store
    print("\n[4/6] Creating vector store...")
    vector_store = initialize_vector_store(embeddings)

    # Create RAG chain
    print("\n[5/6] Creating RAG chain...")
    rag_chain = create_rag_chain(vector_store, llm)
    if rag_chain:
        print("✓ RAG chain created!")
    else:
        print("⚠ RAG chain not available (retrieval-only mode)")

    # Run demo first, then interactive mode (keeps container running)
    print("\n[6/6] Running Hello World demo...")
    demo_mode(vector_store, rag_chain)

    print("\n" + "=" * 60)
    print("Starting interactive mode...")
    print("=" * 60)
    interactive_mode(vector_store, rag_chain)


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


