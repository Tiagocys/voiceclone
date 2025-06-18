from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from TTS.api import TTS
from tempfile import NamedTemporaryFile
import os

app = FastAPI()

# Serve front-end files
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Load TTS model that supports voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

@app.post("/synthesize")
async def synthesize(text: str = Form(...), sample: str = Form(None), voice_upload: UploadFile | None = None):
    speaker_wav = None
    tmp_path = None

    if voice_upload is not None:
        contents = await voice_upload.read()
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
            speaker_wav = tmp_path
    elif sample:
        speaker_wav = os.path.join("public", "voices-models", sample)

    out_path = "output.wav"
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="pt-BR", file_path=out_path)

    if tmp_path is not None:
        os.remove(tmp_path)

    return FileResponse(out_path, media_type="audio/wav", filename="output.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
