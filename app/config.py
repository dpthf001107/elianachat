"""애플리케이션 설정 관리 모듈 입니다.
중앙 설정을 정의해 순환 의존성을 피하고, .env/환경변수를 한 곳에서 읽습니다.
"""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import Field
from pydantic_settings import BaseSettings


def _find_env_file() -> str:
    """환경에 맞는 .env 파일 경로를 찾습니다."""
    # EC2 배포 환경 확인
    if os.path.exists("/home/ubuntu/app/.env"):
        return "/home/ubuntu/app/.env"

    # 로컬 개발 환경: backend/app/.env 확인
    try:
        backend_app_env = Path(__file__).parent / ".env"
        if backend_app_env.exists():
            return str(backend_app_env)
    except Exception:
        pass

    # 프로젝트 루트/.env 확인
    try:
        project_root_env = Path(__file__).parent.parent.parent / ".env"
        if project_root_env.exists():
            return str(project_root_env)
    except Exception:
        pass

    # 기본값: 현재 디렉토리의 .env
    return ".env"


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # 데이터베이스 URL (Neon DB 등 PostgreSQL 연결)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # OpenAI 설정
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # LLM 설정
    llm_provider: str = Field(default="openai")
    local_model_dir: Optional[str] = Field(default=None)

    # 애플리케이션 설정
    app_name: str = "LangChain RAG API"
    app_version: str = "1.0.0"
    debug: bool = False

    class Config:
        """Pydantic 설정."""

        # .env 파일 경로를 동적으로 찾기
        env_file = _find_env_file()
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# .env 파일 수동 로드 (pydantic-settings가 실패할 경우 대비)
def _load_env_manually():
    """수동으로 .env 파일을 로드합니다."""
    env_path = _find_env_file()
    if os.path.exists(env_path):
        print(f"[CONFIG] Loading .env from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # 환경 변수를 무조건 설정 (빈 문자열 덮어쓰기 포함)
                    if key and value:  # 키와 값이 모두 있을 때만
                        os.environ[key] = value
                        print(f"[CONFIG] Set {key}={value[:20]}... from .env file")
    else:
        print(f"[CONFIG] .env file not found at: {env_path}")


# .env 파일 수동 로드 실행
_load_env_manually()

# 전역 설정 인스턴스 (환경 변수가 이미 os.environ에 설정된 후 생성)
settings = Settings()



