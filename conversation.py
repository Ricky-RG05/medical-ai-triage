import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from voice_io import listen, speak
VOICE_MODE = True
# ─────────────────────────────────────────────
# SAME LLM you already use — no second model needed
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama", 
    model="qwen2.5:7b",
    temperature=0.3 #Determines the creativity of the output. 0 for deterministic, higher for more creative responses (0 - 2 range is common).
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT — makes the LLM act as a triage nurse
# ─────────────────────────────────────────────

#Adapt the prompt, so it fills the following: 
# el motivo de consulta, los síntomas y los antecedentes mediante conversación hablada, 
# sin que el paciente tenga que escribir ni llenar formularios

#Los antecedentes son expresados de forma oral, despues de la conversacion, los antecedentes medicos registrados en el sistema son manifestados y unidos con la conversacion habida para generar el reporte final
# ── What the AI needs to collect — in order ──
COLLECTION_FIELDS = {
    "name_age":            "Patient's full name and age",
    "visit_reason":        "Main reason for the visit — in the patient's own words",
    "problem_duration":    "How long has this problem been going on?",
    "evolution":           "Has it been getting better, worse, or staying the same?",
    "associated_factors":  "Does anything make it better or worse?",
    "additional_symptoms": "Any other symptoms the patient associates with the main problem",
    "final_note":          "Anything else the patient wants to add before the doctor reviews their case",
}

TRIAGE_SYSTEM_PROMPT = """You are a medical triage assistant at a Mexican primary care clinic.
Your job is to conduct a warm, open intake interview with a patient in Mexican Spanish.

Collect the following information ONE field at a time, in this exact order:
{fields}

Follow the patient's lead naturally — if they volunteer information for a later field while answering an earlier one, acknowledge it and move forward without asking for it again.

CRITICAL RULES:
- Ask ONE question per turn. No more. Ever.
- If you find yourself writing "además" or "también" — stop. Pick only the most important question.
- Never suggest symptoms the patient has not mentioned.
- Never lead the patient toward any specific condition or diagnosis.
- Never ask about a specific disease unless the patient mentions it explicitly.
- Before asking any question, review the full conversation history:
     -> If a field is already covered, skip it and move to the next uncovered one.
     -> Always move forward — each question must cover new ground.
- Stay warm, calm, and professional. Mexican Spanish only.
- You are "el asistente médico virtual" — NEVER introduce yourself using the patient's name.
- When ALL fields have been collected, respond with EXACTLY this line and nothing else:
  [TRIAGE_COMPLETE]
""".format(
    fields="\n".join(f"- {k}: {v}" for k, v in COLLECTION_FIELDS.items())
)

# ─────────────────────────────────────────────
# Extract structured patient_data from conversation
# ─────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a medical data extractor.
Given a conversation between a triage assistant and a patient,
extract structured patient data and return it as a Python dictionary string.

Extract these fields (use "No especificado" if not mentioned):
- Nombre, Edad, Presión arterial, Altura, Peso, BMI,
- Historial de cáncer, Historial de fumar, Tos crónica,
- Fiebre, Pérdida de peso, Disnea, Dolor torácico, Hemoptisis

Return ONLY a valid Python dict literal. No explanations. No markdown. Example:
{"Nombre": "...", "Edad": "...", ...}"""

def extract_patient_data(conversation_history: list) -> dict:
    """Asks the LLM to parse the conversation into structured patient_data."""
    conversation_text = "\n".join([
        f"{'Paciente' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
        for m in conversation_history
        if not isinstance(m, SystemMessage)
    ])

    extraction_llm = ChatOpenAI(
        base_url="http://localhost:11434/v1", 
        api_key="ollama", 
        model="qwen2.5:7b",
        temperature=0 #Determines the creativity of the output. 0 for deterministic, higher for more creative responses (0 - 2 range is common).
    )

    response = extraction_llm.invoke([
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversación:\n{conversation_text}")
    ])

    try:
        # Safe eval of the dict literal the model returns
        patient_dict = eval(response.content.strip())
        if isinstance(patient_dict, dict):
            return patient_dict
    except Exception:
        pass

    # Fallback: return raw text under a single key so RAG still runs
    return {"Resumen de conversación": conversation_text}


def run_triage_conversation() -> dict:
    """
    Runs the interactive triage conversation in the terminal.
    Returns structured patient_data when complete.
    """
    print("\n" + "="*60)
    print("  SISTEMA DE TRIAJE — Clínica de Primer Nivel")
    print("="*60)
    print("(Escribe 'salir' en cualquier momento para cancelar)\n")

    # Message history — this IS the memory
    #This is a Python type hint, i.e. history is a list of messages.
    history: list = [SystemMessage(content=TRIAGE_SYSTEM_PROMPT)]

    # Opening message from the assistant
    # Opening message from the assistant — with guard against premature TRIAGE_COMPLETE
    opening = llm.invoke(history + [
        HumanMessage(content="El paciente acaba de llegar. Salúdalo cordialmente y pídele que explique su situación. NO uses [TRIAGE_COMPLETE] aquí.")
    ])

    opening_text = opening.content.strip()

    # Strip the tag if the model fired it prematurely
    opening_text = opening_text.replace("[TRIAGE_COMPLETE]", "").strip()

    # Fallback if response ended up empty after stripping
    if not opening_text:
        opening_text = "¡Buenos días! Bienvenido a la clínica. ¿Cómo se siente hoy? Por favor cuénteme qué le trajo por aquí."

    print(f"\n🏥 Asistente: {opening_text}\n")
    if VOICE_MODE:
        speak(opening_text)

    history.append(AIMessage(content=opening_text))

    while True:
        # ── Get patient input ──
        try:
            if VOICE_MODE:
                print("\n👤 Paciente (hablando):")
                user_input = listen(duration=15)
            else:
                user_input = input("👤 Paciente: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n[Sesión cancelada]")
            return {}

        if user_input.lower() in ("salir", "exit", "quit"):
            print("\n[Sesión cancelada por el usuario]")
            return {}

        if not user_input:
            continue

        # Add patient message to history
        history.append(HumanMessage(content=user_input))

        # ── Get LLM response — with retry on empty ──
        MAX_RETRIES: int = 3
        assistant_text = ""

        for attempt in range(MAX_RETRIES):
            response = llm.invoke(history)
            assistant_text = response.content.strip()
            if assistant_text:
                break
            print(f"  ⚠️  Respuesta vacía (intento {attempt + 1}/{MAX_RETRIES}), reintentando...")

        # If still empty after retries, inject a nudge and try one more time
        if not assistant_text:
            nudge = HumanMessage(content="(Por favor continúa con la siguiente pregunta de triaje)")
            response = llm.invoke(history + [nudge])
            assistant_text = response.content.strip()

        # Last resort fallback — should almost never reach this
        if not assistant_text:
            assistant_text = "Disculpe, ¿podría contarme un poco más sobre sus síntomas?"

        # ── Check if triage is complete ──
        if "[TRIAGE_COMPLETE]" in assistant_text:
            print("\n✅ Asistente: Gracias, tengo toda la información necesaria.")
            print("             Procesando su evaluación clínica...\n")
            history.append(AIMessage(content=assistant_text))
            break

        # Strip the tag if it leaked into a normal response
        assistant_text = assistant_text.replace("[TRIAGE_COMPLETE]", "").strip()

        # Final empty check after stripping
        if not assistant_text:
            assistant_text = "Disculpe, ¿podría contarme un poco más sobre sus síntomas?"

        # Normal conversational response
        print(f"\n🏥 Asistente: {assistant_text}\n")
        if VOICE_MODE:
            speak(assistant_text)
            
        history.append(AIMessage(content=assistant_text))

    # ── Extract structured data from the full conversation ──
    print("🔍 Extrayendo datos clínicos de la conversación...")

    transcript = ""

    for m in history:
        if not isinstance(m, SystemMessage):
            if isinstance(m, HumanMessage):
                transcript += f"Paciente: {m.content}\n"
            else:
                transcript += f"Asistente: {m.content}\n"

    transcript = transcript.strip()

    patient_data = extract_patient_data(history)

    return patient_data, transcript

# ─────────────────────────────────────────────
# CONDITION CLASSIFIER — picks the right PDF folder/s
# ─────────────────────────────────────────────

FOLDERS = {
    "diabetes_tipo2":                    "Diabetes, Glucosa elevada, sed excesiva, fatiga, visión borrosa, poliuria, diabetes, control glicémico",
    "diarrea_aguda":                     "Diarrea, evacuaciones líquidas, dolor abdominal, náuseas, vómito, gastroenteritis, deshidratación",
    "dislipidemias_hipercolesterolemia": "Colesterol alto, triglicéridos, lípidos en sangre, riesgo cardiovascular, dislipidemia",
    "faringoamigdalitis_aguda":          "Dolor de garganta, amígdalas inflamadas, fiebre, dificultad al tragar, faringitis, amigdalitis",
    "hipertension_arterial":             "Presión arterial alta, hipertensión, dolor de cabeza, mareos, control de presión",
    "infeccion_urinaria_mujer":          "Ardor al orinar, frecuencia urinaria, dolor pélvico, infección urinaria, cistitis, disuria",
    "infeccion_vias_respiratorias":      "Tos, catarro, resfriado, congestión nasal, moco, rinorrea, dolor de cabeza, gripe leve",
    "influenza_n1h1":                    "Influenza, gripe fuerte, fiebre alta súbita, dolor muscular intenso, malestar general severo, H1N1",
    "lumbalgia_aguda_cronica":           "Dolor de espalda baja, lumbalgia, dolor al doblar, ciática, dolor lumbar, espalda",
}

CLASSIFICATION_PROMPT = f"""You are a medical triage classifier working at a Mexican primary care clinic.
Your job is to read a patient conversation and select the most appropriate clinical guidelines.

Available guidelines:
{chr(10).join(f'- "{k}": {v}' for k, v in FOLDERS.items())}

RULES:
- Select ONLY guidelines that are DIRECTLY relevant to the patient's complaints.
- If the patient has ONE clear main complaint → return ONLY that one guideline.
- If the patient has TWO OR MORE clearly distinct conditions → return multiple guidelines.
- NEVER return more than 3 guidelines.
- NEVER guess or add guidelines not clearly supported by what the patient said.
- Base your decision ONLY on what the patient explicitly mentioned.

Reply with ONLY the folder name(s), one per line, exactly as written above.
No explanations. No extra text. Example of single:
faringoamigdalitis_aguda

Example of multiple:
diabetes_tipo2
hipertension_arterial"""


def classify_condition(transcript: str) -> list[str]:
    """
    Returns a LIST of folder names that match the patient's conversation.
    Usually just one, but can be multiple for complex cases.
    """
    classifier_llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0
    )

    response = classifier_llm.invoke([
        SystemMessage(content=CLASSIFICATION_PROMPT),
        HumanMessage(content=f"Conversación:\n{transcript}")
    ])

    raw = response.content.strip()

    # Parse — one folder per line
    candidates = [line.strip() for line in raw.splitlines() if line.strip()]

    # Validate — keep only known folders, ignore hallucinations
    valid = [c for c in candidates if c in FOLDERS]

    if not valid:
        #Get a general folder, if the model didn't recognize any specific classification
        print(f"⚠️  Clasificación no reconocida ('{raw}'). Usando 'infeccion_vias_respiratorias' por defecto.")
        return ["infeccion_vias_respiratorias"]

    print(f"📂 Guías clínicas seleccionadas: {', '.join(valid)}")
    return valid

