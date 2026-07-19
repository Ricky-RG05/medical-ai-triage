import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PDF_BASE_DIR = os.path.join(BASE_DIR, "PDFs")
TEST_FOLDER  = "diabetes_tipo2"

pdf_path     = os.path.join(PDF_BASE_DIR, TEST_FOLDER, "718GER (1).pdf")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434",
)

# ── Extract text with pdfplumber ──
print("📄 Extrayendo texto con pdfplumber...")
with pdfplumber.open(pdf_path) as pdf:
    full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

print(f"✅ Caracteres extraídos: {len(full_text)}")
print(f"\n📝 Primeros 500 caracteres:\n{full_text[:500]}")

# ── Split into chunks ──
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
docs   = [Document(page_content=full_text, metadata={"source": pdf_path})]
splits = text_splitter.split_documents(docs)
print(f"\n📦 Chunks creados: {len(splits)}")

# ── Build a temporary in-memory vectorstore ──
print("\n🔄 Construyendo vectorstore de prueba...")
vs        = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 4})
print(f"✅ Vectorstore listo. Chunks: {vs._collection.count()}")

# ── Run the same test questions ──
"""TEST_QUESTIONS = [
    "¿Qué porcentaje de pacientes con diarrea aguda presentan tres o más evacuaciones?",
    "¿Qué antibiótico se recomienda para Shigella sp?",
    "¿En qué porcentaje reduce el lavado de manos el riesgo de diarrea?",
    "¿Qué porcentaje de pérdida de peso debe haber para recomendar soluciones salinas isotónicas?",
    "¿Qué tipo de alimentos se deben recomendar en caso de tratamiento no farmacológico?",
]"""

TEST_QUESTIONS = [
    "¿Cuál es la dosis recomendada y efectos adversos de tomar el medicamento ACARBOSA?",
    "En qué condiciones se debe iniciar insulina NPH en el caso de Dx reciente DM tipo 2?",
    "En qué consiste una terapia dual?"
]

print("\n" + "="*60)
print("  🧪 RAG RETRIEVAL TEST — diarrea_aguda (pdfplumber)")
print("="*60)

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n❓ Test {i}: {question}")
    docs = retriever.invoke(question)
    print(f"📄 Retrieved {len(docs)} chunks:")
    for j, doc in enumerate(docs, 1):
        print(f"\n  Chunk {j}: {doc.page_content[:300]}")
    print("-"*40)

    from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── LLM ──
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen2.5:7b",
    temperature=0,
    max_tokens=150  # force short answers
)

SYSTEM_PROMPT = """You are a medical assistant. Answer each question using ONLY the context provided.
Answer in ONE sentence maximum. If the answer is not in the context, say exactly: "No encontrado en la guía."
Do not use prior knowledge. Only what the context says."""

print("\n" + "="*60)
print("  🤖 LLM ANSWER TEST — diarrea_aguda")
print("="*60)

for i, question in enumerate(TEST_QUESTIONS, 1):
    # Retrieve chunks
    chunks = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in chunks)

    # Ask the LLM
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ])

    print(f"\n❓ {question}")
    print(f"🤖 {response.content.strip()}")
    print("-"*40)