import os
import glob
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PDF_BASE_DIR    = os.path.join(BASE_DIR, "PDFs")
CHROMA_BASE_DIR = os.path.join(BASE_DIR, "chroma_db")


def get_or_build_vectorstore(folder_name: str, embeddings) -> Chroma | None:
    chroma_path = os.path.join(CHROMA_BASE_DIR, folder_name)
    pdf_folder  = os.path.join(PDF_BASE_DIR, folder_name)
    pdf_files   = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    if not pdf_files:
        print(f"  ❌ No PDF found in: {pdf_folder}")
        return None

    if os.path.exists(chroma_path) and os.listdir(chroma_path):
        print(f"  ⚡ Embeddings cached — loading: {folder_name}")
        return Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

    print(f"  🔄 Building embeddings for: {folder_name}")
    print(f"     PDF: {os.path.basename(pdf_files[0])}")

    loader = PyPDFLoader(pdf_files[0])
    docs   = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=chroma_path
    )

    print(f"  ✅ Embeddings saved: {chroma_path}")
    return vectorstore


def preprocess_all_pdfs(embeddings) -> None:
    print("\n" + "="*60)
    print("  📦  VERIFICACIÓN DE EMBEDDINGS")
    print("="*60)

    os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

    pdf_folders = [
        f for f in os.listdir(PDF_BASE_DIR)
        if os.path.isdir(os.path.join(PDF_BASE_DIR, f))
    ]

    built   = 0
    skipped = 0

    for folder_name in sorted(pdf_folders):
        chroma_path = os.path.join(CHROMA_BASE_DIR, folder_name)
        if os.path.exists(chroma_path) and os.listdir(chroma_path):
            print(f"  ✅ {folder_name} — cached")
            skipped += 1
        else:
            result = get_or_build_vectorstore(folder_name, embeddings)
            if result:
                built += 1
            else:
                print(f"  ⚠️  {folder_name} — no PDF found, skipped")

    print(f"\n  Built: {built}  |  Already cached: {skipped}")
    print("="*60 + "\n")


def build_retriever(selected_folders: list, embeddings):
    """
    Loads vectorstores for selected folders and returns a single retriever.
    For multiple folders, combines results manually — no MergerRetriever needed.
    """
    vectorstores = []
    for folder in selected_folders:
        vs = get_or_build_vectorstore(folder, embeddings)
        if vs:
            vectorstores.append(vs)

    if not vectorstores:
        return None

    if len(vectorstores) == 1:
        return vectorstores[0].as_retriever(search_kwargs={"k": 4})

    # ── Multiple guides — combine manually ──
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from typing import List, Any

    class CombinedRetriever(BaseRetriever):
        stores: List[Any]

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            results = []
            seen = set()
            for vs in self.stores:
                docs = vs.similarity_search(query, k=4)
                for doc in docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        results.append(doc)
            return results[:8]

    return CombinedRetriever(stores=vectorstores)