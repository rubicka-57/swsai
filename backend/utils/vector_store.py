from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import get_settings


def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store() -> Chroma:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
