import os
import json
import tempfile
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("firebase_init")
logger.setLevel(logging.INFO)

import firebase_admin
from firebase_admin import credentials, db

def ensure_firebase_initialized():
    if firebase_admin._apps:
        return

    cred_json = os.environ.get("FIREBASE_CRED_JSON")
    cred_path = os.environ.get("FIREBASE_CRED_PATH")

    if cred_json and not cred_path:
        tf = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        tf.write(cred_json)
        tf.flush()
        cred_path = tf.name
        logger.info("Wrote FIREBASE_CRED_JSON to temp file")

    if cred_path:
        cred = credentials.Certificate(cred_path)
    else:
        required = ["private_key", "private_key_id", "client_email", "client_id"]
        if not all(os.environ.get(k) for k in required):
            raise RuntimeError("Firebase credential not provided. Set FIREBASE_CRED_JSON or required env vars.")
        cred_obj = {
            "type": "service_account",
            "project_id": "bigcon-2025",
            "private_key_id": os.environ["private_key_id"],
            "private_key": os.environ["private_key"].replace("\\n", "\n"),
            "client_email": os.environ["client_email"],
            "client_id": os.environ["client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40bigcon-2025.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com",
        }
        cred = credentials.Certificate(cred_obj)

    database_url = "https://bigcon-2025-default-rtdb.asia-southeast1.firebasedatabase.app"
    firebase_admin.initialize_app(cred, {"databaseURL": database_url})
    logger.info("Firebase initialized, DB URL=%s", database_url)

def get_db_ref(path: str):
    ensure_firebase_initialized()
    return db.reference(path)