# asr_ser_api.py
from fastapi import FastAPI, UploadFile
import torchaudio, speechbrain as sb
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

app = FastAPI()

# ASR – Whisper-large-v3 (ou qualquer modelo HF)
asr_proc   = AutoProcessor.from_pretrained("openai/whisper-large-v3")
asr_model  = AutoModelForSeq2SeqLM.from_pretrained("openai/whisper-large-v3")

# SER – wav2vec2-IEMOCAP (SpeechBrain)
ser_model = sb.pretrained.interfaces.classification.ClfMixin.from_hparams(
    "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    wav, sr = torchaudio.load(file.file)
    # 1. transcrição
    input_features = asr_proc(wav.squeeze(), sampling_rate=sr, return_tensors="pt").input_features
    pred_ids = asr_model.generate(input_features)
    text = asr_proc.batch_decode(pred_ids, skip_special_tokens=True)[0]
    # 2. emoção
    emo = ser_model.classify_batch(wav)[0]
    return {"text": text, "emotion": emo}
