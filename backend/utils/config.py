from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True)

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    documents_dir: Path = Field(default=BASE_DIR / "documents", alias="DOCUMENTS_DIR")
    chroma_dir: Path = Field(default=BASE_DIR / "chroma_db", alias="CHROMA_DIR")
    collection_name: str = Field(default="company_documents", alias="COLLECTION_NAME")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")

    def model_post_init(self, __context: object) -> None:
        if not self.documents_dir.is_absolute():
            self.documents_dir = BASE_DIR / self.documents_dir
        if not self.chroma_dir.is_absolute():
            self.chroma_dir = BASE_DIR / self.chroma_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
