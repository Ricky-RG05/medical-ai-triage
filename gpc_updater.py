import os
import json
import hashlib
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
#To open file to be ignored by git: "code .gitignore" → add the file name to the list of ignored files → save and close the file. Now git will ignore it.
#All files starting with a dot (.) are hidden files in Windows, so you won't see it in the file explorer unless you enable "Show hidden files" in the view options.
#Or... unless you run in the terminal "ls -Force"

# ─────────────────────────────────────────────
# GOOGLE DRIVE CONFIG
# ─────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdrive_credentials.json")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ─────────────────────────────────────────────
# REGISTRY — maps local folder name to the Google Drive FOLDER ID
# Whatever PDF is inside that Drive folder gets downloaded automatically
# ─────────────────────────────────────────────
GPC_REGISTRY = {
    "faringoamigdalitis_aguda":{"code": "IMSS-073-08", "drive_folder_id": "1bPqzq1c7xDuyLvMGyuFP2tTQcz0zpdf8"},
    "infeccion_vias_respiratorias": {"code": "IMSS-062-08", "drive_folder_id": "1d0TNJVFkzxZgvCdSPM4INc6vd2Ogjaez"},
    "diabetes_tipo2":          {"code": "IMSS-718-18", "drive_folder_id": "1wyXPNA7HnI-Wsj7mAqHGTWPuxywU8GXV"},
    "hipertension_arterial":   {"code": "IMSS-076-08", "drive_folder_id": "14kodBqvxFTa9ynRrhztkIdg_7wv-qZhM"},
    "infeccion_urinaria_mujer":{"code": "IMSS-077-08", "drive_folder_id": "1G8Ok_4q4UNoXeosj1-eJkIT4N5CkXnSn"},
    "influenza_n1h1":          {"code": "IMSS-000-08", "drive_folder_id": "1c7B98BfvMt9uN-XWIZYvxA_uCYnYcW6o"},
    "lumbalgia_aguda_cronica": {"code": "IMSS-045-08", "drive_folder_id": "17iOoV12-e47A2-7KloFyvwKraLylvbhI"},
    "dislipidemias_hipercolesterolemia": {"code": "IMSS-233-09", "drive_folder_id": "1OFCsW1SduzCDXXtrEewnchLFVTUr-bBF"},
    "diarrea_aguda": {"code": "SSA-106-08", "drive_folder_id": "1LEcbBdtmFgXRnO_Lhr7w3KShpPeOEj81"},
}

def should_run_monthly_update() -> bool:
    """Returns True if more than 30 days have passed since last update check."""
    versions = _load_versions()
    last_check = versions.get("_last_check_date")

    if last_check is None:
        return True  # Never run before → run now

    last_check_date = datetime.fromisoformat(last_check)
    days_since = (datetime.now() - last_check_date).days

    print(f"  📅 Última revisión: {last_check_date.strftime('%d/%m/%Y')} ({days_since} días atrás)")
    return days_since >= 30

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
#print(__file__) returns relative path; hence os.path.abspath to get the absolute one!
PDF_BASE_DIR  = os.path.join(BASE_DIR, "PDFs")
VERSIONS_FILE = os.path.join(PDF_BASE_DIR, "versions.json")

def _get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)
# ---> Give me a remote control for the "Drive" API, version 3, authenticated as this service account (identified with the credentials-object!)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    #To open() the file in binary mode and read it in chunks to avoid loading the entire file into memory
    with open(path, "rb") as f:
        #Run iter() until f.read() returns b"" (empty bytes), reading 8192 bytes at a time
        for chunk in iter(lambda: f.read(8192), b""):
            #Feed each 8192-byte chunk into the hash function
            h.update(chunk)
    return h.hexdigest()
    #Converts the final fingerprint into a readable string of letters and numbers 


def _load_versions() -> dict:
    if os.path.exists(VERSIONS_FILE):
        #To deserialize the JSON data from the file into a Python dictionary, we use json.load()
        with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
            #json.load() maps the JSON structure to standard Python types, i.e. into a readbale dictionary
    return {}


def _save_versions(versions: dict) -> None:
    with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)
        #Python dict  →  json.dump()  →  JSON text file on disk
        #JSON file    →  json.load()  →  Python dict back in memory


def _find_pdf_in_folder(service, folder_id: str):
    """
    Searches a Google Drive folder and returns the first PDF found.
    Returns (file_id, file_name) or (None, None) if no PDF exists.
    """
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",   # newest PDF first, in case there's more than one
        pageSize=5
    ).execute()

    files = results.get("files", [])
    if not files:
        return None, None

    # Take the most recently modified PDF in the folder
    newest = files[0]
    return newest["id"], newest["name"]


def _download_file(service, file_id: str, dest_path: str) -> bool:
    try:
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        with open(dest_path, "rb") as f:
            if not f.read(4) == b"%PDF":
                print(f"    ⚠️  Archivo descargado no es un PDF válido")
                return False
        return True

    except Exception as e:
        print(f"    ⚠️  Error descargando: {e}")
        return False


def update_all_guides(force: bool = False) -> dict:
    print("\n" + "="*60)
    print("  📥  ACTUALIZADOR DE GUÍAS CLÍNICAS (Google Drive)")
    print("="*60)

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"  ❌ No se encontró el archivo de credenciales: {SERVICE_ACCOUNT_FILE}")
        return {}

    service  = _get_drive_service()
    versions = _load_versions()
    summary  = {}

    for folder_name, guide in GPC_REGISTRY.items():
        code      = guide["code"]
        folder_id = guide["drive_folder_id"]
        print(f"\n  📋 {code} → {folder_name}/")

        local_folder = os.path.join(PDF_BASE_DIR, folder_name)
        os.makedirs(local_folder, exist_ok=True)

        # ── Find whatever PDF is currently in the Drive folder ──
        print(f"    🔍 Buscando PDF en carpeta de Drive...")
        file_id, file_name = _find_pdf_in_folder(service, folder_id)

        if file_id is None:
            print(f"    ❌  No se encontró ningún PDF en la carpeta de Drive")
            # Keep existing local file if there is one
            existing = [f for f in os.listdir(local_folder) if f.endswith(".pdf")]
            summary[folder_name] = "kept_existing" if existing else "failed"
            continue

        print(f"    📄 Encontrado: {file_name}")

        dest_path = os.path.join(local_folder, file_name)
        tmp_path  = dest_path + ".tmp"

        if not _download_file(service, file_id, tmp_path):
            existing = [f for f in os.listdir(local_folder) if f.endswith(".pdf")]
            summary[folder_name] = "kept_existing" if existing else "failed"
            continue

        new_hash = _sha256(tmp_path)
        old_hash = versions.get(folder_name, {}).get("hash")
        old_name = versions.get(folder_name, {}).get("file_name")

        if not force and old_hash == new_hash and old_name == file_name:
            os.remove(tmp_path)
            print(f"    ✅ Sin cambios — versión actual vigente")
            summary[folder_name] = "unchanged"
        else:
            # ── Clean up: remove ANY old PDFs in this folder before saving new one ──
            for old_file in os.listdir(local_folder):
                if old_file.endswith(".pdf") and old_file != file_name:
                    os.remove(os.path.join(local_folder, old_file))

            os.replace(tmp_path, dest_path)
            versions[folder_name] = {
                "code": code,
                "file_name": file_name,
                "hash": new_hash,
                "source": "google_drive",
                "last_updated": datetime.now().isoformat(),
            }
            _save_versions(versions)
            action = "actualizada" if old_hash else "descargada por primera vez"
            print(f"    ✅ Guía {action}: {file_name}")
            summary[folder_name] = "updated"

    print("\n" + "="*60)
    updated   = sum(1 for s in summary.values() if s == "updated")
    unchanged = sum(1 for s in summary.values() if s == "unchanged")
    failed    = sum(1 for s in summary.values() if s == "failed")
    print(f"  ✅ Actualizadas: {updated}  |  ⏸️  Sin cambios: {unchanged}  |  ❌ Fallidas: {failed}")
    print("="*60 + "\n")

    # ── Save the date of this check ──
    versions = _load_versions()
    # ── Save the date of this check into versions.json ──
    versions["_last_check_date"] = datetime.now().isoformat()
    _save_versions(versions)  # versions dict is already in memory — just save it
    #If key (_last_check_date) hasn't yet been added, it simply gets added on the fly. If it already exists, it gets identified (O(1), it's a dictionary, it's ultra fast!) and the value gets overriden!!
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Actualizador de Guías Clínicas (Google Drive)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    update_all_guides(force=args.force)