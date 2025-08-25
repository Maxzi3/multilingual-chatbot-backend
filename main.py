import os
import logging
import json
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------- Load Env ----------------
load_dotenv()

SUPPORTED_LANGS = {
    "en": "English",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "sw": "Swahili",
    "de": "German"
}

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "500"))
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

# Enhanced debugging for API key
print("="*50)
print("🔍 HTTP API CONFIGURATION")
print("="*50)

if not QWEN_API_KEY:
    print("❌ ERROR: No API key found in environment variables")
    print("📝 Make sure your .env file contains: DASHSCOPE_API_KEY=your-key-here")
    raise RuntimeError("Qwen API key not set. Please set DASHSCOPE_API_KEY in .env")
else:
    print(f"✅ API key found: {QWEN_API_KEY[:10]}...{QWEN_API_KEY[-4:]} (length: {len(QWEN_API_KEY)})")
    print(f"🌐 API endpoint: {QWEN_API_URL}")

print("="*50)

# ---------------- Enhanced Logger ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbot")

# ---------------- FastAPI ----------------
app = FastAPI(
    title="Multilingual Chatbot with Qwen (HTTP API)",
    description="A chatbot that uses Alibaba Qwen models via HTTP API and supports multiple languages",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    language: str
    conversation_history: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    language: str
    language_name: str

class LanguagesResponse(BaseModel):
    supported_languages: dict
    total_count: int

class HealthResponse(BaseModel):
    status: str
    message: str
    supported_languages: dict

class DebugResponse(BaseModel):
    api_key_status: str
    api_key_length: int
    api_endpoint: str
    test_result: str
    error_details: Optional[str] = None

# ---------------- HTTP API Integration ----------------
def make_qwen_request(messages: list, model: str = "qwen-plus", temperature: float = 0.7, max_tokens: int = 200) -> dict:
    """Make HTTP request to Qwen API"""
    
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    logger.info(f"📤 Making HTTP request to Qwen API")
    logger.info(f"🔑 Using API key: {QWEN_API_KEY[:10]}...{QWEN_API_KEY[-4:]}")
    logger.info(f"📋 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            QWEN_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        logger.info(f"📥 Response status: {response.status_code}")
        
        # Try to parse JSON response
        try:
            response_data = response.json()
            logger.info(f"📊 Response data: {json.dumps(response_data, indent=2)}")
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse JSON. Raw response: {response.text}")
            raise Exception(f"Invalid JSON response: {response.text}")
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response_data
            }
        else:
            error_info = response_data.get('error', {})
            error_message = error_info.get('message', 'Unknown error')
            error_code = error_info.get('code', response.status_code)
            error_type = error_info.get('type', 'Unknown')
            
            logger.error(f"❌ API Error: {error_code} - {error_message} (Type: {error_type})")
            
            return {
                "success": False,
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "type": error_type,
                    "status_code": response.status_code
                }
            }
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request timeout")
        return {
            "success": False,
            "error": {
                "code": "timeout",
                "message": "Request to Qwen API timed out",
                "type": "timeout_error"
            }
        }
    except requests.exceptions.ConnectionError:
        logger.error("❌ Connection error")
        return {
            "success": False,
            "error": {
                "code": "connection_error",
                "message": "Failed to connect to Qwen API",
                "type": "connection_error"
            }
        }
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return {
            "success": False,
            "error": {
                "code": "unexpected_error",
                "message": str(e),
                "type": type(e).__name__
            }
        }

def test_api_connection() -> dict:
    """Test the API connection with a simple request"""
    logger.info("🧪 Testing API connection...")
    
    test_messages = [{"role": "user", "content": "Hello! Please respond with 'API test successful'"}]
    result = make_qwen_request(test_messages, max_tokens=20, temperature=0.1)
    
    if result["success"]:
        response_text = ""
        if "choices" in result["data"] and result["data"]["choices"]:
            response_text = result["data"]["choices"][0].get("message", {}).get("content", "")
        
        return {
            "status": "success",
            "message": "API connection successful",
            "response": response_text
        }
    else:
        return {
            "status": "error",
            "message": f"API test failed: {result['error']['message']}",
            "error": result["error"]
        }

def generate_reply(prompt: str, conversation_history: str = None) -> str:
    """Generate reply using HTTP API"""
    try:
        logger.info(f"📝 Generating reply for prompt: '{prompt[:50]}...'")
        
        messages = []
        if conversation_history:
            messages.append({"role": "system", "content": conversation_history})
        messages.append({"role": "user", "content": prompt})
        
        result = make_qwen_request(messages)
        
        if result["success"]:
            response_data = result["data"]
            if "choices" in response_data and response_data["choices"]:
                reply_text = response_data["choices"][0].get("message", {}).get("content", "").strip()
                if reply_text:
                    logger.info(f"✅ Successfully generated reply: '{reply_text[:100]}...'")
                    return reply_text
                else:
                    raise Exception("Empty response from API")
            else:
                raise Exception("No choices in API response")
        else:
            error = result["error"]
            error_msg = f"{error['message']} (Code: {error['code']}, Type: {error['type']})"
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"💥 Error in generate_reply: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

# ---------------- Endpoints ----------------
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Multilingual Chatbot API using Qwen HTTP API",
        "version": "2.0.0",
        "api_method": "HTTP requests",
        "endpoints": {
            "chat": "/chat",
            "languages": "/languages",
            "health": "/health",
            "debug": "/debug",
            "test-api": "/test-api",
            "docs": "/docs"
        },
        "example_request": {
            "message": "Hello, how are you?",
            "language": "en"
        }
    }

@app.get("/debug", response_model=DebugResponse)
async def debug_info():
    """Debug endpoint to check API configuration"""
    logger.info("🔍 Debug endpoint called")
    
    api_key_status = "found" if QWEN_API_KEY else "missing"
    api_key_length = len(QWEN_API_KEY) if QWEN_API_KEY else 0
    
    # Test API connection
    test_result = test_api_connection()
    
    return DebugResponse(
        api_key_status=api_key_status,
        api_key_length=api_key_length,
        api_endpoint=QWEN_API_URL,
        test_result=test_result["status"],
        error_details=json.dumps(test_result, indent=2)
    )

@app.get("/test-api")
async def test_api():
    """Test API connection endpoint"""
    logger.info("🧪 API test endpoint called")
    result = test_api_connection()
    
    if result["status"] == "success":
        return {"status": "success", "message": "API is working correctly", "details": result}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"API test failed: {result.get('message', 'Unknown error')}"
        )

@app.get("/languages", response_model=LanguagesResponse)
async def get_supported_languages():
    return LanguagesResponse(
        supported_languages=SUPPORTED_LANGS,
        total_count=len(SUPPORTED_LANGS)
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Test API as part of health check
    test_result = test_api_connection()
    status = "healthy" if test_result["status"] == "success" else "unhealthy"
    message = test_result.get("message", "Unknown status")
    
    return HealthResponse(
        status=status,
        message=f"Qwen HTTP API status: {message}",
        supported_languages=SUPPORTED_LANGS
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint using HTTP API"""
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        user_language = request.language.lower()
        if user_language not in SUPPORTED_LANGS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language code: {user_language}. Supported languages: {list(SUPPORTED_LANGS.keys())}"
            )
        
        language_name = SUPPORTED_LANGS[user_language]
        logger.info(f"💬 Processing message in {language_name} ({user_language}): {user_message}")
        
        reply = generate_reply(user_message, request.conversation_history)
        
        return ChatResponse(
            reply=reply,
            language=user_language,
            language_name=language_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# ---------------- Convenience Endpoints ----------------
@app.post("/chat/english")
async def chat_english(request: dict):
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    return await chat(ChatRequest(message=message, language="en"))

@app.post("/chat/yoruba")
async def chat_yoruba(request: dict):
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    return await chat(ChatRequest(message=message, language="yo"))

@app.post("/chat/igbo")
async def chat_igbo(request: dict):
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    return await chat(ChatRequest(message=message, language="ig"))

@app.post("/chat/hausa")
async def chat_hausa(request: dict):
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    return await chat(ChatRequest(message=message, language="ha"))

# ---------------- Startup Event ----------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Multilingual Chatbot API (HTTP version)...")
    logger.info("🧪 Running startup API test...")
    
    test_result = test_api_connection()
    if test_result["status"] == "success":
        logger.info("✅ HTTP API connection test passed - ready to serve requests!")
    else:
        logger.error("❌ HTTP API connection test failed - check your configuration")
        logger.error(f"❌ Error details: {json.dumps(test_result, indent=2)}")

# ---------------- Main Runner ----------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 Starting HTTP API server on {host}:{port}")
    print(f"📚 Visit http://{host}:{port}/docs for API documentation")
    print(f"🔍 Visit http://{host}:{port}/debug for debugging info")
    print(f"🧪 Visit http://{host}:{port}/test-api to test API connection")
    
    uvicorn.run("main:app", host=host, port=port, reload=True)