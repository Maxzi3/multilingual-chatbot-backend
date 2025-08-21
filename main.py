from fastapi import FastAPI
import requests
import os

app = FastAPI()

# ✅ Get your HF API key from Hugging Face
HF_API_KEY = os.getenv("HF_API_KEY")  # keep it safe in Render's Environment Variables
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

@app.get("/")
def home():
    return {"message": "Multilingual backend is running!"}

@app.post("/translate")
async def translate(text: str, src: str, tgt: str):
    """
    Example: {"text": "Bawo ni", "src": "yo", "tgt": "en"}
    """
    model = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    url = f"https://api-inference.huggingface.co/models/{model}"

    response = requests.post(url, headers=headers, json={"inputs": text})
    return response.json()
