"""Backend router package."""

try:
    from app.router import chat_router  # EC2
except ImportError:
    from backend.app.router import chat_router  # Local

__all__ = ["chat_router"]
