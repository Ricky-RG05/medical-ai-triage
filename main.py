import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from embedding_generator import preprocess_all_pdfs, build_retriever
from langchain_ollama import OllamaEmbeddings
import glob
from gpc_updater import update_all_guides
# Import custom modules for conversation and PDF generation
from report_generator import generate_pdf
from conversation import run_triage_conversation, classify_condition

#------ Run at startup — skips download if guides haven't changed
#Store the current system's date twice, once when the comparison was made and the current one. Make a comparison between both to see if a month has gone by. If so, run update_all_guides() (only to be run once a month, because it takes a bit to run!) 
from gpc_updater import update_all_guides, should_run_monthly_update

if should_run_monthly_update():
    print("🔄 Han pasado 30 días — actualizando guías clínicas...")
    update_all_guides()
else:
    print("✅ Guías clínicas vigentes — no requieren actualización.")

"""
Make gpc_updater.py run once in a month to prove all pdfs haven't been changed lately! 
"""

"""
Script the download of ollama and all required libraries in a batch file!
"""

#Current dimension is 768, i.e. the default for nomic-embed-text-v2-moe, i.e. the highest dimension available, and honestly, more than enough! 
# ── Define embeddings ──
embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434",
)

# ── Pre-process any missing PDFs at startup ──
preprocess_all_pdfs(embeddings)

# ── Run triage conversation ──
patient_data, conversation_transcript = run_triage_conversation()

if not conversation_transcript:
    print("No se recopiló información del paciente. Saliendo.")
    exit()

# ── Classify condition ──
selected_folders = classify_condition(conversation_transcript)

# ── Build retriever from selected folders (uses cache) ──
retriever = build_retriever(selected_folders, embeddings)

if not retriever:
    print("❌ No guides available. Exiting.")
    exit()

# ── Human-readable labels per condition ──
CONDITION_LABELS = {
    "diabetes_tipo2":                    ("Diabetes Mellitus Tipo 2",              "¿Requiere este paciente ajuste de tratamiento o referencia? Evalúa según la GPC."),
    "diarrea_aguda":                     ("Enfermedad Diarreica Aguda",             "¿Debería este paciente recibir tratamiento o ser referido? Evalúa según la GPC."),
    "dislipidemias_hipercolesterolemia": ("Dislipidemias e Hipercolesterolemia",    "¿Requiere este paciente intervención o referencia? Evalúa su perfil lipídico según la GPC."),
    "faringoamigdalitis_aguda":          ("Faringoamigdalitis Aguda",               "¿Requiere este paciente antibiótico o referencia? Evalúa según la GPC."),
    "hipertension_arterial":             ("Hipertensión Arterial",                  "¿Requiere este paciente ajuste de tratamiento o referencia urgente? Evalúa según la GPC."),
    "infeccion_urinaria_mujer":          ("Infección de Vías Urinarias",            "¿Debería este paciente recibir tratamiento o ser referido? Evalúa según la GPC."),
    "infeccion_vias_respiratorias":      ("Infección Aguda de Vías Respiratorias",  "¿Requiere este paciente tratamiento o referencia? Evalúa según la GPC."),
    "influenza_n1h1":                    ("Influenza AH1N1",                        "¿Debería este paciente ser referido o aislado? Evalúa según la GPC."),
    "lumbalgia_aguda_cronica":           ("Lumbalgia Aguda y Crónica",              "¿Requiere este paciente tratamiento especializado o referencia? Evalúa según la GPC."),
}

condition_title, question = CONDITION_LABELS.get(
    selected_folders[0],
    ("Evaluación Clínica General", "¿Debería este paciente ser referido a un especialista según la GPC?")
)

print(f"\n📋 Datos recopilados: {len(conversation_transcript)} campos")

# ─────────────────────────────────────────────
# PATIENT DATA — edit this block as needed
# ─────────────────────────────────────────────
"""This line will be replaced by the actual patient data collected during the conversation. The structure of the patient_data dictionary should be consistent with the keys expected in the prompt template, and the values should be formatted in a way that is clear and informative for the LLM to process. You can modify the keys and values based on the specific data points you collect from the patient during the triage conversation."""

#Only as dummy to test the triage system running! 
# ─────────────────────────────────────────────
# TEST PATIENTS — for prompt version evaluation
# Switch by changing: patient_data = TEST_PATIENTS["carlos"]
# ─────────────────────────────────────────────
    # TEST 1 — 
    # Condición esperada: faringoamigdalitis_aguda
    # Qué decir en la entrevista:
    #   - Me llamo Carlos Mendoza, tengo 28 años
    #   - Desde hace 3 días me duele mucho la garganta, especialmente al tragar
    #   - También tengo fiebre desde ayer, como de 38.5°C
    #   - Me siento muy cansado y no tengo ganas de comer
    #   - No tengo tos ni mocos, solo el dolor de garganta y la fiebre
    #   - (si te pregunta algo más) No, creo que eso es todo

 # TEST 2 — 
    # Condición esperada: hipertension_arterial + dislipidemias_hipercolesterolemia
    # (comorbilidad — dos condiciones simultáneas, buen test para el clasificador)
    # Qué decir en la entrevista:
    #   - Me llamo Esperanza Villanueva, tengo 61 años
    #   - Vengo porque últimamente me dan muchos dolores de cabeza, sobre todo en la mañana
    #   - También me mareo a veces cuando me levanto rápido
    #   - Mi doctor anterior me dijo que tenía el colesterol alto pero lo dejé de tratar
    #   - No hago mucho ejercicio y como bastante grasa
    #   - (si te pregunta) No tomo ningún medicamento ahorita
    #   - (al final) Sí, también a veces siento que me late muy fuerte el corazón

  # TEST 3 — 
    # Condición esperada: infeccion_urinaria_mujer
    # Qué decir en la entrevista:
    #   - Me llamo Guadalupe Ramírez, tengo 34 años
    #   - Llevo como dos días con mucho ardor cuando orino
    #   - Voy al baño muy seguido pero casi no sale nada
    #   - También siento una presión rara aquí abajo (zona pélvica)
    #   - No tengo fiebre ni escalofríos
    #   - (si te pregunta) No, no tengo dolor de espalda ni en los riñones
    #   - El ardor empeora al final de cada vez que orino

TEST_PATIENTS = {

    "test1": {
        "Sexo": "Masculino",
        "Edad": "28 años",
        "Presión arterial": "118/74 mmHg",
        "Peso": "72 kg",
        "Talla": "173 cm",
        "Glucosa": "88 mg/dL"
    },

   
    "test2": {
        "Sexo": "Femenino",
        "Edad": "61 años",
        "Presión arterial": "158/96 mmHg",
        "Peso": "78 kg",
        "Talla": "158 cm",
        "Glucosa": "102 mg/dL"
    },

  
    "test3": {
        "Sexo": "Femenino",
        "Edad": "34 años",
        "Presión arterial": "112/70 mmHg",
        "Peso": "63 kg",
        "Talla": "162 cm",
        "Glucosa": "85 mg/dL"
    },

}

# ── Active test patient — change this line to switch ──
patient_data = TEST_PATIENTS["test3"]


# ─────────────────────────────────────────────
# 1. Load & index the PDF guideline
# ─────────────────────────────────────────────
"""
Worth flagging for future-you: when you eventually persist Chroma to disk for performance, you'll need to delete the persisted index any time you change embedding models. 
"""

#As it can be cleary seen, the offload from the embeddings is directly to the CPU, i.e. since it requires so little resources, it's better to offload it to the CPU and free up GPU resources for the LLM inference, which is the most resource-intensive part of the process. This way, we can optimize the overall performance of the system by ensuring that the GPU is primarily dedicated to running the LLM, while the CPU handles the less demanding task of generating embeddings for the document chunks.

""" Worth flagging! 

The search document doesn't run automatically by Ollama, so it must be manually managed. Just consider it when chaiging the 
actual pipeline, if it'll run all PDFs in RAG or only the selected one, etc.

# For embedding your PDF chunks at storage time
doc_embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434",
)

# For embedding queries at retrieval time  
query_embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe",
    base_url="http://localhost:11434",
)

And then wrap your texts before embedding:
python# When building the vectorstore
prefixed_splits = [
    Document(
        page_content="search_document: " + doc.page_content,
        metadata=doc.metadata
    ) 
    for doc in splits
]

vectorstore = Chroma.from_documents(
    documents=prefixed_splits, 
    embedding=doc_embeddings
)

# When querying
def retrieve(query: str):
    prefixed_query = "search_query: " + query
    return vectorstore.similarity_search(prefixed_query, k=4)

"""

"""
2. Your k=4 flexibility comment — valid concern, here's the real solution
Instead of hardcoding k=4, make it proportional to how many documents you have:
pythonimport math

def get_retriever(vectorstore, num_docs: int):
    # Scale k with collection size, but cap it sensibly
    k = min(max(4, math.ceil(num_docs * 0.1)), 20)
    return vectorstore.as_retriever(search_kwargs={"k": k})
"""

# ─────────────────────────────────────────────
# 2. LLM via LM Studio
# ─────────────────────────────────────────────

#Standarized interface for creating an LLM based on the ChatOpenAI interface: 
#Documentation for it can be found here: https://docs.langchain.com/oss/python/integrations/chat/openai
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama", 
    model="qwen2.5:7b",
    temperature=0, #Determines the creativity of the output. 0 for deterministic, higher for more creative responses (0 - 2 range is common).
    max_tokens=2048
)

# ─────────────────────────────────────────────
# 3. Prompt template
# ─────────────────────────────────────────────

#ChatPromptTemplate allows us to create a structured prompt with multiple input variables (context, question, patient data) and a clear format for the LLM to follow.
#Consider that they're multiple ways of structuring the template per-se, but this one was used, because it allows us to make the whole template using a single string-format, which then is converted into a prompt object that can be used in the Runnable chain, once we use the conversion "from_template".
template = """Eres un asistente médico siguiendo las guías clínicas oficiales mexicanas (GPC SSA-022-08).

INSTRUCCIÓN CRÍTICA: Debes responder OBLIGATORIAMENTE con las siguientes 4 secciones numeradas. 
Cada sección debe tener mínimo 3-5 oraciones detalladas. NO des respuestas cortas.

=== CONTEXTO DE LA GUÍA CLÍNICA ===
{context}

=== DATOS DEL PACIENTE ===
{patient_data}

=== TRANSCRIPCIÓN DE LA CONVERSACIÓN ===
{conversation}

=== PREGUNTA ===
{question}

Responde EXACTAMENTE con este formato, sin omitir ninguna sección:

1. Evaluación de Riesgo
[Analiza detalladamente el nivel de riesgo del paciente basándote en sus síntomas, edad, historial y factores de riesgo según la GPC. Mínimo 4 oraciones.]

2. Síntomas Relevantes Detectados
[Lista y explica cada síntoma relevante que presenta el paciente y su importancia clínica según la guía. Mínimo 4 oraciones.]

3. Estudios Recomendados
[Especifica exactamente qué estudios diagnósticos recomienda la GPC para este caso y por qué. Mínimo 3 oraciones.]

4. Conclusión y Criterio de Referencia
[Concluye si el paciente debe ser referido, a qué especialista, con qué urgencia, y qué seguimiento se recomienda según la GPC. Mínimo 4 oraciones.]"""

#Official documentation for ChatPromptTemplate can be found here: https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)

# Format patient data as a readable string for the prompt
patient_data_str = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in patient_data.items()])

#Instead of simply writing in the final code: llm.invoke(), we rather create a Runnable chain that allows us to have more control over the flow of data and the processing steps. This way, we can easily modify or extend the chain in the future if needed, without having to rewrite the entire logic. The Runnable chain also makes it clearer how the different components (retriever, prompt, llm) interact with each other and how the patient data is integrated into the process.
chain = (
    {
        #Input dictionary
        "context": retriever,
        "question": RunnablePassthrough(),
        "patient_data": RunnableLambda(lambda _: patient_data_str),
        "conversation": RunnableLambda(lambda _: conversation_transcript)
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────
# 4. Run the chain
# ─────────────────────────────────────────────

# ── Show exactly what will be used for the analysis ──
print("\n" + "="*60)
for folder in selected_folders:
    pdf_files = glob.glob(os.path.join(r"PDFs", folder, "*.pdf"))
    if pdf_files:
        print(f"  📂 {folder} → {os.path.basename(pdf_files[0])}")
        print(f"  📁 RUTA COMPLETA             : {pdf_files[0]}")
print("="*60 + "\n")

print("🔍 Analizando datos del paciente con la guía clínica...")

result = chain.invoke(question)
print("\n✅ Análisis completado. Generando reporte PDF...\n")

# ─────────────────────────────────────────────
# 5. Generate the PDF report
# ─────────────────────────────────────────────
generate_pdf(
    patient_data=patient_data,
    analysis_result=result,
    condition_title=condition_title,
    selected_folder=selected_folders[0]    # primary condition
)

"""
Final for installer script for clinic machines: 

1. Install Ollama
2. ollama pull nomic-embed-text-v2-moe
3. ollama create medical-nemotron -f Modelfile
4. Set OLLAMA_KEEP_ALIVE=-1
5. Run our Python project

"""