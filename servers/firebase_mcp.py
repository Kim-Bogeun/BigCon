import os
import json
import random
from dotenv import load_dotenv
from typing import Any, Iterable, List
from collections import OrderedDict

import firebase_admin
from firebase_admin import credentials, db
from mcp.server.fastmcp import FastMCP

load_dotenv()

if not firebase_admin._apps:

    cred_json = OrderedDict()
    cred_json["type"] = "service_account"
    cred_json["project_id"] = "bigcon-2025"
    cred_json["private_key_id"] = os.getenv("private_key_id")
    cred_json["private_key"] = os.getenv("private_key").replace('\\n', '\n')
    cred_json["client_email"] = os.getenv("client_email")
    cred_json["client_id"] = os.getenv("client_id")
    cred_json["auth_uri"] = "https://accounts.google.com/o/oauth2/auth"
    cred_json["token_uri"] = "https://oauth2.googleapis.com/token"
    cred_json["auth_provider_x509_cert_url"] = "https://www.googleapis.com/oauth2/v1/certs"
    cred_json["client_x509_cert_url"] = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40bigcon-2025.iam.gserviceaccount.com"
    cred_json["universe_domain"] = "googleapis.com"
    

    dump_json = json.dumps(cred_json)
    load_json = json.loads(dump_json)
    cred = credentials.Certificate(load_json)
    # cred = credentials.Certificate('bigcon-2025-firebase-adminsdk-fbsvc-4409b3177b.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://bigcon-2025-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

mcp = FastMCP(
    name="Firebase",
    instructions="A Retriever that can retrieve information from the Firebase Realtime Database.",
    host="0.0.0.0"
)



def _find_encoded_mcts_by_name(mct_nm: str) -> list[str]:
    """
    Return list of ENCODED_MCT keys that match the given MCT_NM (case-insensitive).
    """
    try:
        ref = db.reference("/가맹점_개요정보")
        data = ref.get()
        if not isinstance(data, dict):
            return []
        encoded_list: list[str] = []
        for key, record in data.items():
            if not isinstance(record, dict):
                continue
            name = record.get("MCT_NM") or record.get("mct_nm")
            if name and str(name).lower() == mct_nm.lower():
                encoded = record.get("ENCODED_MCT") or key
                encoded_list.append(str(encoded))
        return encoded_list
    except Exception:
        return []
    
@mcp.tool()
async def search_franchise_by_name(mct_nm: str) -> dict[str, Any]:
    """
    Searches for franchise overview information in database by MCT_NM (franchise name).
    """
    print(f"search_franchise_by_name called with mct_nm: {mct_nm}")
    try:
        ref = db.reference("/신한은행_데이터")
        data = ref.get()
        if not isinstance(data, dict):
            return []
        
        record = data.get(mct_nm)
        return record
    except Exception as e:
        print(f"Error in search_franchise_by_name: {e}")
        return {"error": str(e)}

@mcp.tool()
async def marketing_channels_info(q_type: str) -> List[dict[str, Any]]:
    """
    Get information of marketing channels from Firebase Realtime Database.

    Args:
        q_type (str): Query type
    Returns:
        List[dict[str, Any]]: A list of marketing channel information, each with 'name' and 'data' fields
    """
    ref = db.reference("/marketing_channels")
    data = ref.get() 

    if not data:
        return []

    result = [{"name": name, "data": info} for name, info in data.items()]
    return result
    
@mcp.tool()
async def marketing_method_info(q_type: str) -> List[dict[str, Any]]:
    """
    Get information of online marketing methods from Firebase Realtime Database.

    Args:
        q_type (str): Query type
    Returns:
        List[dict[str, Any]]: A list of 3 randomly selected marketing channel information, each with 'name' and 'data' fields
    """
    ref = db.reference("/online_marketing")
    data = ref.get() 

    if not data:
        return []

    # 랜덤으로 3개 선택
    sampled_items = random.sample(list(data.items()), 3)
    result = [{"name": name, "data": info} for name, info in sampled_items]
    return result

if __name__ == "__main__":
    print("Starting Firebase MCP server...")
    mcp.run(
        # transport="streamable-http"
        transport="sse"
    )
