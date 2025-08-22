# --- Imports ---
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect, DetectorFactory

# Set a deterministic seed for langdetect for consistent results
DetectorFactory.seed = 42

# ---------------- Language Codes ----------------
# A dictionary of supported language codes and their full names
SUPPORTED_LANGS = {
    "en": "english",
    "yo": "yoruba",
    "ig": "igbo",
    "ha": "hausa",
}

# LibreTranslate endpoint (can use self-hosted or free demo)
# This uses an environment variable for the URL, falling back to the public demo
LT_URL = os.getenv("LT_URL", "https://libretranslate.de/translate")

# ---------------- Helper Functions ----------------

def detect_lang(text: str) -> str:
    """
    Detects the language of a given text.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        str: The detected language code (e.g., "en", "yo"). Defaults to "en" if detection fails or is unsupported.
    """
    try:
        code = detect(text)
        return code if code in SUPPORTED_LANGS else "en"
    except Exception:
        # Fallback to English if language detection fails
        return "en"

def translate(text: str, source: str, target: str) -> str:
    """
    Translates text from a source language to a target language using LibreTranslate API.
    
    Args:
        text (str): The text to translate.
        source (str): The source language code.
        target (str): The target language code.
        
    Returns:
        str: The translated text.
        
    Raises:
        HTTPException: If the translation API call fails.
    """
    # If source and target are the same, no translation is needed
    if source == target:
        return text
    
    try:
        # Make a POST request to the LibreTranslate API
        resp = requests.post(
            LT_URL,
            json={"q": text, "source": source, "target": target, "format": "text"},
            timeout=10,
        )
        # Raise an exception for bad status codes (4xx or 5xx)
        resp.raise_for_status()
        return resp.json()["translatedText"]
    except Exception as e:
        # Catch any errors and raise an HTTPException for FastAPI to handle
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# This is the user's provided function, now integrated into the app.
def generate_reply(prompt: str) -> str:
    """
    Generates a text reply using the Hugging Face Inference API.
    
    Args:
        prompt (str): The prompt for the text generation model.
        
    Returns:
        str: The generated text reply.
        
    Raises:
        HTTPException: If the Hugging Face API call fails or the API key is missing.
    """
    # Get the API key from environment variables
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face API token is not set.")

    url = "https://api-inference.huggingface.co/models/distilgpt2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 60}}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data[0]["generated_text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")


# ---------------- FastAPI Application ----------------

app = FastAPI(
    title="Multilingual Chatbot",
    description="A simple chatbot that translates user messages, generates an English reply with Hugging Face, and translates the response back to the user's language."
)

class ChatRequest(BaseModel):
    """Pydantic model for the request body of the /chat endpoint."""
    message: str

class ChatResponse(BaseModel):
    """Pydantic model for the response body of the /chat endpoint."""
    reply: str
    detected_lang: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    The main chat endpoint.
    
    Processes a user's message through the following steps:
    1. Detects the user's language.
    2. Translates the message to English.
    3. Generates a reply in English using the Hugging Face API.
    4. Translates the reply back to the user's original language.
    """
    user_text = req.message

    # 1. Detect user language
    src = detect_lang(user_text)

    # 2. Translate to English if needed
    en_text = translate(user_text, source=src, target="en")

    # 3. Generate English reply using Hugging Face
    en_reply = generate_reply(en_text)

    # 4. Translate back to user language
    final_reply = translate(en_reply, source="en", target=src)

    # Return the final reply and the detected language
    return ChatResponse(reply=final_reply, detected_lang=src)

# ---------------- Local run ----------------
# This block allows you to run the application locally for testing
if __name__ == "__main__":
    import uvicorn
    # Make sure to run the command 'uvicorn app:app --reload'
    uvicorn.run(app, host="0.0.0.0", port=8000)


