from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_settings
from utils.vector_store import get_vector_store


SUPPORTED_EXTENSIONS = {".pdf"}


def find_document_files(documents_dir: Path) -> list[Path]:
    documents_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_pdf(pdf_path: Path) -> list[Document]:
    pages = PyPDFLoader(str(pdf_path)).load()

    for page_number, page in enumerate(pages, start=1):
        page.metadata.update(
            {
                "source": pdf_path.name,
                "source_path": str(pdf_path),
                "page": page_number,
            }
        )

    return pages


def load_documents(document_files: list[Path]) -> list[Document]:
    documents: list[Document] = []

    for document_file in document_files:
        if document_file.suffix.lower() == ".pdf":
            documents.extend(load_pdf(document_file))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    source_counts: dict[tuple[str, int], int] = {}
    chunk_ids: list[str] = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "document")
        page = int(chunk.metadata.get("page", 0))
        key = (source, page)
        chunk_index = source_counts.get(key, 0)
        source_counts[key] = chunk_index + 1

        chunk.metadata["chunk"] = chunk_index
        chunk_ids.append(f"{source}:page-{page}:chunk-{chunk_index}")

    return chunk_ids


def clear_vector_store(vector_store: Any) -> int:
    existing = vector_store.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    return len(existing_ids)


def ingest_documents() -> dict[str, Any]:
    settings = get_settings()
    document_files = find_document_files(settings.documents_dir)

    if not document_files:
        return {
            "message": "No supported documents found.",
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_replaced": 0,
            "sources": [],
        }

    documents = load_documents(document_files)
    chunks = split_documents(documents)
    chunk_ids = build_chunk_ids(chunks)

    vector_store = get_vector_store()
    replaced_count = clear_vector_store(vector_store)

    if chunks:
        vector_store.add_documents(chunks, ids=chunk_ids)

    return {
        "message": "Documents ingested successfully.",
        "documents_processed": len(document_files),
        "chunks_created": len(chunks),
        "chunks_replaced": replaced_count,
        "sources": [path.name for path in document_files],
    }


if __name__ == "__main__":
    print(ingest_documents())
