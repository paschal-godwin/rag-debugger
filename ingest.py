from typing import List
from pypdf import PdfReader
from langchain.schema import Document

def load_pdfs_from_folder(folder: str) -> List[Document]:
    """
    Loads PDFs and returns LangChain Documents with metadata: source + page.
    """
    docs: List[Document] = []
    import os

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder, filename)
        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": i + 1
                    }
                )
            )
    return docs
