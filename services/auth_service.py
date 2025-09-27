"""
Authentication service for HuggingFace Hub integration.
"""
import os
import json
from typing import Optional
from huggingface_hub import HfApi
from core.logging_config import get_logger

logger = get_logger(__name__)

TOKEN_FILE = 'hf_token.json'


def load_login_token() -> Optional[str]:
    """
    저장된 HuggingFace 토큰을 로드합니다.

    Returns:
        Optional[str]: 토큰 또는 None
    """
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                data = json.load(f)
                token = data.get('token')
                if token:
                    logger.info("HuggingFace token loaded from file")
                    return token
    except Exception as e:
        logger.error(f"Failed to load token: {e}")

    return None


def save_login_token(token: str) -> bool:
    """
    HuggingFace 토큰을 저장합니다.

    Args:
        token: HuggingFace API 토큰

    Returns:
        bool: 저장 성공 여부
    """
    try:
        with open(TOKEN_FILE, 'w') as f:
            json.dump({'token': token}, f)
        logger.info("HuggingFace token saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        return False


def delete_login_token() -> bool:
    """
    저장된 토큰을 삭제합니다.

    Returns:
        bool: 삭제 성공 여부
    """
    try:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            logger.info("HuggingFace token deleted")
        return True
    except Exception as e:
        logger.error(f"Failed to delete token: {e}")
        return False


def validate_token(token: str) -> bool:
    """
    토큰의 유효성을 검증합니다.

    Args:
        token: 검증할 토큰

    Returns:
        bool: 토큰 유효성
    """
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        return user_info is not None
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return False


def get_user_info(token: Optional[str] = None) -> Optional[dict]:
    """
    사용자 정보를 가져옵니다.

    Args:
        token: HuggingFace API 토큰 (None이면 저장된 토큰 사용)

    Returns:
        Optional[dict]: 사용자 정보 또는 None
    """
    if token is None:
        token = load_login_token()

    if not token:
        return None

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        logger.info(f"Retrieved user info for: {user_info.get('name', 'unknown')}")
        return user_info
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        return None


def login_with_token(token: str) -> tuple[bool, str]:
    """
    토큰을 사용하여 로그인합니다.

    Args:
        token: HuggingFace API 토큰

    Returns:
        tuple[bool, str]: (성공 여부, 메시지)
    """
    if not token or not token.strip():
        return False, "토큰이 입력되지 않았습니다."

    token = token.strip()

    # 토큰 유효성 검증
    if not validate_token(token):
        return False, "유효하지 않은 토큰입니다."

    # 토큰 저장
    if not save_login_token(token):
        return False, "토큰 저장에 실패했습니다."

    # 사용자 정보 가져오기
    user_info = get_user_info(token)
    if user_info:
        username = user_info.get('name', 'Unknown')
        return True, f"성공적으로 로그인했습니다: {username}"
    else:
        return False, "사용자 정보를 가져올 수 없습니다."


def logout() -> tuple[bool, str]:
    """
    로그아웃합니다.

    Returns:
        tuple[bool, str]: (성공 여부, 메시지)
    """
    try:
        if delete_login_token():
            return True, "성공적으로 로그아웃했습니다."
        else:
            return False, "로그아웃 처리 중 오류가 발생했습니다."
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        return False, f"로그아웃 실패: {str(e)}"


def is_logged_in() -> bool:
    """
    현재 로그인 상태를 확인합니다.

    Returns:
        bool: 로그인 여부
    """
    token = load_login_token()
    if not token:
        return False

    return validate_token(token)


def get_current_user() -> Optional[str]:
    """
    현재 로그인된 사용자 이름을 반환합니다.

    Returns:
        Optional[str]: 사용자 이름 또는 None
    """
    user_info = get_user_info()
    if user_info:
        return user_info.get('name')
    return None