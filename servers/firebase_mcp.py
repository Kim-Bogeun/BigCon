import os
import json
from dotenv import load_dotenv
from typing import Any, Iterable
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

# ref1_map = {
#     "ENCODED_MCT": "가맹점구분번호",
#     "MCT_BSE_AR": "가맹점주소",
#     "MCT_NM": "가맹점명",
#     "MCT_BRD_NUM": "브랜드구분코드",
#     "MCT_SIGUNGU_NM": "가맹점지역",
#     "HPSN_MCT_ZCD_NM": "업종",
#     "HPSN_MCT_BZN_CD_NM": "상권",
#     "ARE_D": "개설일",
#     "MCT_ME_D": "폐업일"
# }

# ref2_map = {
#     "ENCODED_MCT": "가맹점구분번호",
#     "TA_YM": "기준년월",
#     "MCT_OPE_MS_CN": "가맹점 운영개월수 구간",
#     "RC_M1_SAA": "매출금액 구간",
#     "RC_M1_TO_UE_CT": "매출건수 구간",
#     "RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
#     "RC_M1_AV_NP_AT": "객단가 구간",
#     "APV_CE_RAT": "취소율 구간",
#     "DLV_SAA_RAT": "배달매출금액 비율",
#     "M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
#     "M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
#     "M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
#     "M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
#     "M12_SME_RY_ME_MCT_RAT": "동일 업종 내 해지 가맹점 비중",
#     "M12_SME_BZN_ME_MCT_RAT": "동일 상권 내 해지 가맹점 비중"
# }

# ref3_map = {
#     "ENCODED_MCT": "가맹점구분번호",
#     "TA_YM": "기준년월",
#     "M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
#     "M12_MAL_30_RAT": "남성 30대 고객 비중",
#     "M12_MAL_40_RAT": "남성 40대 고객 비중",
#     "M12_MAL_50_RAT": "남성 50대 고객 비중",
#     "M12_MAL_60_RAT": "남성 60대이상 고객 비중",
#     "M12_FME_1020_RAT": "여성 20대이하 고객 비중",
#     "M12_FME_30_RAT": "여성 30대 고객 비중",
#     "M12_FME_40_RAT": "여성 40대 고객 비중",
#     "M12_FME_50_RAT": "여성 50대 고객 비중",
#     "M12_FME_60_RAT": "여성 60대이상 고객 비중",
#     "MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
#     "MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
#     "RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
#     "RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
#     "RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율"
# }

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

if __name__ == "__main__":
    print("Starting Firebase MCP server...")
    mcp.run(
        # transport="streamable-http"
        transport="sse"
    )
