"""
FastAPI middleware for request processing and validation.

This module contains middleware components for the FastAPI server,
including JSON repair and request processing utilities.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json
import re
import logging
import time
from typing import Dict, Any


class JSONRepairMiddleware(BaseHTTPMiddleware):
    """JSON 요청 자동 복구 미들웨어"""

    def __init__(self, app, logger=None):
        """Initialize JSON repair middleware."""
        super().__init__(app)
        self.logger = logger or logging.getLogger("JSONRepair")

    async def dispatch(self, request: Request, call_next):
        """Process request with JSON repair if needed."""
        # POST 요청이고 JSON 컨텐츠 타입인 경우만 처리
        if request.method == "POST" and request.headers.get("content-type", "").startswith("application/json"):
            try:
                # 원본 body 읽기
                body = await request.body()
                original_body = body.decode('utf-8')

                # JSON 파싱 시도
                try:
                    json.loads(original_body)
                    # 파싱 성공하면 그대로 진행
                except json.JSONDecodeError:
                    # JSON 오류 시 자동 복구 시도
                    repaired_body = self.repair_json(original_body)
                    if repaired_body != original_body:
                        self.logger.info(f"JSON 자동 복구 적용: {original_body[:50]}... -> {repaired_body[:50]}...")
                        # 수정된 body로 새로운 Request 객체 생성
                        request._body = repaired_body.encode('utf-8')
                    else:
                        # JSON 복구도 실패한 경우 텍스트 추출 시도
                        fallback_data = self.extract_text_fallback(original_body)
                        if fallback_data["text"]:
                            self.logger.info(f"텍스트 Fallback 적용: {fallback_data}")
                            request._body = json.dumps(fallback_data).encode('utf-8')

            except Exception as e:
                self.logger.warning(f"JSON 복구 중 오류: {e}")

        response = await call_next(request)
        return response

    def repair_json(self, json_str: str) -> str:
        """JSON 문자열 자동 복구"""
        try:
            # 1. 백슬래시 이스케이프 문제 수정
            # \\" -> " 변환 (잘못된 이스케이프)
            repaired = re.sub(r'\\+"', '"', json_str)

            # 2. 단일 따옴표를 이중 따옴표로 변환
            repaired = re.sub(r"'([^']*)':", r'"\1":', repaired)
            repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)

            # 3. 키에 따옴표 누락 수정
            repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)

            # 4. 후행 쉼표 제거
            repaired = re.sub(r',\s*}', '}', repaired)
            repaired = re.sub(r',\s*]', ']', repaired)

            # 5. 개행문자 이스케이프 처리
            repaired = repaired.replace('\\n', '\\\\n').replace('\\r', '\\\\r')

            # 6. 최종 검증
            json.loads(repaired)
            return repaired

        except Exception:
            # 복구 실패 시 원본 반환
            return json_str

    def extract_text_fallback(self, json_str: str) -> Dict[str, Any]:
        """JSON 파싱 완전 실패 시 텍스트 추출"""
        try:
            # text 필드 추출 시도
            text_match = re.search(r'"?text"?\s*:\s*"([^"]*)"', json_str)
            if text_match:
                return {"text": text_match.group(1)}

            # 단일 따옴표로 된 text 필드 추출
            text_match = re.search(r"'?text'?\s*:\s*'([^']*)'", json_str)
            if text_match:
                return {"text": text_match.group(1)}

        except Exception:
            pass

        return {"text": ""}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    def __init__(self, app, logger=None):
        """Initialize request logging middleware."""
        super().__init__(app)
        self.logger = logger or logging.getLogger("RequestLogger")

    async def dispatch(self, request: Request, call_next):
        """Log request information."""
        start_time = time.time()

        # Log request
        self.logger.info(f"{request.method} {request.url.path}")

        response = await call_next(request)

        # Log response time
        process_time = time.time() - start_time
        self.logger.info(f"Request processed in {process_time:.3f}s - Status: {response.status_code}")

        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Simple CORS middleware for development."""

    def __init__(self, app, allow_origins=None, allow_methods=None, allow_headers=None):
        """Initialize CORS middleware."""
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: Request, call_next):
        """Add CORS headers to response."""
        if request.method == "OPTIONS":
            # Handle preflight requests
            response = Response(status_code=200)
        else:
            response = await call_next(request)

        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = ",".join(self.allow_origins)
        response.headers["Access-Control-Allow-Methods"] = ",".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ",".join(self.allow_headers)

        return response