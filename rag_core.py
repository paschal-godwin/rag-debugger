from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

INDEX_PATH = "data/indexes/faiss_index"

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_faiss_index(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise ValueError(
            "No chunks were created. This usually means no text was extracted from the PDFs "
            "(common with scanned/image PDFs). Try a text-based PDF or add OCR."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.from_documents(chunks, embeddings)

def save_index(vs: FAISS) -> None:
    vs.save_local(INDEX_PATH)

def load_index() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieve_with_scores(vs: FAISS, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Returns [(doc, score)] where score is FAISS distance (lower is typically better).
    """
    return vs.similarity_search_with_score(query, k=k)

def answer_with_citations(query: str, retrieved: List[Tuple[Document, float]]) -> Dict[str, Any]:
    """
    Generates an answer using retrieved context and returns:
    { answer: str, citations: [ {source, page, chunk_preview} ] }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Build a compact context block
    context_blocks = []
    citations = []
    for idx, (doc, score) in enumerate(retrieved, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        snippet = doc.page_content[:450].replace("\n", " ").strip()
        context_blocks.append(f"[{idx}] (source={src}, page={page}, score={score})\n{doc.page_content}\n")
        citations.append({
            "ref": idx,
            "source": src,
            "page": page,
            "score": score,
            "preview": snippet
        })

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a careful assistant. Answer the user's question ONLY using the context below.
If the context does not contain enough information, say: "Not enough information in the provided documents."

Return:
1) A clear answer
2) A short "Sources used" list referencing the bracket numbers like [1], [2]

Question:
{query}

Context:
{context}
""".strip()

    resp = llm.invoke(prompt)
    return {"answer": resp.content, "citations": citations, "context_used": context_blocks}
