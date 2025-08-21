from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# ---------- Language Maps ----------
NLLB_CODES = {
    "en": "eng_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
}

# ---------- Load models once ----------
chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# ---------- Utils ----------
def translate(text: str, src: str, tgt: str) -> str:
    translator_tokenizer.src_lang = src
    inputs = translator_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(tgt),
            max_new_tokens=128
        )
    return translator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Request Schema ----------
class Message(BaseModel):
    text: str
    lang: str  # "en", "yo", "ig", "ha"

# ---------- Routes ----------
@app.post("/chat")
def chat(msg: Message):
    src_code = NLLB_CODES.get(msg.lang, "eng_Latn")

    # translate to English if needed
    if src_code != "eng_Latn":
        english_text = translate(msg.text, src_code, "eng_Latn")
    else:
        english_text = msg.text

    # generate reply
    reply_en = chatbot(english_text, max_length=100, do_sample=True)[0]["generated_text"]

    # translate back
    if src_code != "eng_Latn":
        reply_final = translate(reply_en, "eng_Latn", src_code)
    else:
        reply_final = reply_en

    return {"reply": reply_final}
