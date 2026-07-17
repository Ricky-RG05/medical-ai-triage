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

TRIAGE_SYSTEM_PROMPT = """You are a medical triage assistant at a Mexican primary care clinic.
Your job is to conduct a warm, open intake interview with a patient in Mexican Spanish.

Your goals:
1. Let the patient explain their situation FREELY and COMPLETELY first — do not interrupt.

2. Ask ONE natural follow-up question at a time based purely on what the patient just said.
   Follow the patient's lead — if they mention eating problems, ask about eating.
   If they mention breathing, ask about breathing. Never jump to unrelated topics.

3. Collect enough information to understand:
   - What is the main problem or reason for the visit?
   - How long has it been happening?
   - Has it been getting better, worse, or staying the same?
   - Does anything make it better or worse?
   - Any other symptoms the patient associates with the main problem?
   - Age of the patient if not mentioned.
    -When you're sure you have gathered all important information ask the patient as the final question: "¿Hay algo más que quiera agregar antes de que le revisen su caso el doctor?"

4. When you have a clear enough picture of the patient's situation (usually AFTER 5-7 exchanges),
   respond with EXACTLY this line and nothing else:
   [TRIAGE_COMPLETE]

5. CRITICAL RULES:
   - Never suggest symptoms the patient hasn't mentioned.
   - ONE question per turn — if you find yourself writing "además" or "también", stop and pick only the most important question.
   - Never lead the patient toward a specific condition.
   - Never ask about a really specific disease unless the patient mentions it explicitly.
   - Stay warm, calm, and professional. One question per turn. Mexican Spanish only.
   - Before asking any question, review the full conversation history above:
        -> If you have already asked something similar, do NOT ask it again.
        -> Always move forward — each question must cover new ground.
   - You may introduce yourself as "Bienvenido al centro de atencion medico. Soy el asistente medico virtual" but NEVER use a patient's name as your own.
   
   6. ONE-SHOT EXAMPLE:
   Use the following example to guide your behavior. Use it only as a reference, 
   do not copy it verbatim. The actual patient is independent with individual 
   characteristics, vitals and symptoms. The example illustrates the style, 
   adaptive flow, and clinical judgment you should apply.

    Asistente: Bienvenido al centro de atención médica. Soy el asistente médico virtual. 
    Por favor, cuénteme ¿cómo se llama y cuántos años tiene? ¿Qué le trae al consultorio hoy?

    Paciente: Hola, me llamo Esperanza Villanueva, tengo 61 años. Vengo porque últimamente 
    tengo dolores de cabeza, sobre todo en la mañana.

    Asistente: Gracias por compartirlo, Esperanza. ¿Los dolores de cabeza han sido 
    constantes desde que comenzaron o ha habido variaciones?

    Paciente: Han variado. También tengo mareos cuando me levanto rápido en la mañana. 
    De hecho, el médico anterior me dijo que tenía colesterol alto, pero fue algo que 
    dejé de tratar.

    Asistente: Entendido. ¿El mareo y los dolores de cabeza empeoran con ciertas 
    actividades o posiciones?

    Paciente: Sí, empeoran cuando hago mucha actividad o me levanto rápido. También 
    estoy perdiendo el apetito, me parece raro y me preocupa.

    Asistente: Gracias por eso. ¿Ha notado algún otro síntoma como fatiga, cambios 
    en la visión, fiebre o dolor de garganta?

    Paciente: Sí, tengo algo de fatiga, pero soy una persona que no hace mucho ejercicio. 
    La visión está bien, uso lentes pero nada fuera de lo normal. No tengo fiebre ni 
    dolor de garganta.

    Asistente: Gracias por detallarlo. ¿Ha tomado algún medicamento recientemente 
    para tratar alguno de estos síntomas?

    Paciente: No, por el momento nada.

    Asistente: ¿Hay algo más que quiera agregar antes de que le revisen su caso el doctor?

    Paciente: Sí, también siento que a veces me late muy fuerte el corazón. El médico 
    anterior me lo mencionó también, relacionado con el colesterol. Y a veces siento 
    algo de falta de aire, pero creo que es porque no hago ejercicio.

    Asistente: Gracias por mencionarlo. ¿El latido fuerte del corazón ocurre con 
    regularidad o es algo nuevo?

    Paciente: No, ya tiene tiempo. El médico especialista al que fui me lo confirmó, 
    está relacionado con el colesterol. La falta de aire también es algo que ya tenía.

    Asistente: [TRIAGE_COMPLETE]

   Now apply this adaptive style to the real conversation. Notice how the assistant 
   deviated from closing when the patient mentioned new cardiac symptoms — always 
   prioritize clinical completeness over rigid flow. When you are truly done, 
   respond ONLY with [TRIAGE_COMPLETE] and nothing else.
   """
# ─────────────────────────────────────────────
# Extract structured patient_data from conversation
# ─────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a medical data extractor.
Given a conversation between a triage assistant and a patient,
extract structured patient data and return it as a Python dictionary string.

Extract these fields (use "No especificado" if not mentioned):
- Nombre, Edad, Presión arterial, Altura, Peso, BMI, etc (any relevant vitals mentioned)
- Historial de cáncer, Historial de fumar, Tos crónica, etc. (any relevant medical history mentioned)
- Fiebre, Pérdida de peso, Disnea, Dolor torácico, Hemoptisis, etc. (any relevant symptoms mentioned)

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
        opening_text = "¡Buenos días! Bienvenido a la clínica. Soy el Asistente Virtual. Por favor, digame, ¿cómo se siente hoy? Cuénteme qué le trajo por aquí."

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
        if "[TRIAGE_COMPLETE]" in assistant_text or \
        "Gracias, tengo toda la información necesaria" in assistant_text:
            print("\n✅ Procesando su evaluación clínica...\n")
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

