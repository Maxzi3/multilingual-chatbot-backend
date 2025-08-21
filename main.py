from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# --------------------------
# Load translation models
# --------------------------
def load_translator(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Yoruba
yo_en_model, yo_en_tokenizer = load_translator("yo", "en")
en_yo_model, en_yo_tokenizer = load_translator("en", "yo")

# Igbo
ig_en_model, ig_en_tokenizer = load_translator("ig", "en")
en_ig_model, en_ig_tokenizer = load_translator("en", "ig")

# Hausa
ha_en_model, ha_en_tokenizer = load_translator("ha", "en")
en_ha_model, en_ha_tokenizer = load_translator("en", "ha")

# --------------------------
# Load chatbot (English only)
# --------------------------
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# --------------------------
# Request schema
# --------------------------
class ChatRequest(BaseModel):
    text: str
    lang: str  # "yo", "ig", "ha", or "en"

# --------------------------
# Helper functions
# --------------------------
def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs, max_length=200)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def chat_response(text):
    inputs = chat_tokenizer.encode(text + chat_tokenizer.eos_token, return_tensors="pt")
    outputs = chat_model.generate(inputs, max_length=200, pad_token_id=chat_tokenizer.eos_token_id)
    return chat_tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# --------------------------
# API endpoint
# --------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_text, lang = req.text, req.lang.lower()

        # Step 1: Translate to English if needed
        if lang == "yo":
            user_text = translate(user_text, yo_en_model, yo_en_tokenizer)
        elif lang == "ig":
            user_text = translate(user_text, ig_en_model, ig_en_tokenizer)
        elif lang == "ha":
            user_text = translate(user_text, ha_en_model, ha_en_tokenizer)

        # Step 2: Get chatbot response in English
        reply_en = chat_response(user_text)

        # Step 3: Translate back to user language
        if lang == "yo":
            reply = translate(reply_en, en_yo_model, en_yo_tokenizer)
        elif lang == "ig":
            reply = translate(reply_en, en_ig_model, en_ig_tokenizer)
        elif lang == "ha":
            reply = translate(reply_en, en_ha_model, en_ha_tokenizer)
        else:
            reply = reply_en

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
