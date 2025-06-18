# clone.py
from TTS.api import TTS

# 1) Carrega o modelo
#    “your_tts” já engloba TTS+vocoder.
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# 2) Sintetiza usando os dois samples:
wav = tts.tts(
    text="Olá, esta é uma demonstração final de clonagem de voz!",
    # timbre: 10–20 s de fala neutra
    speaker_wav="Neutro.wav",
    # estilo/prosódia: 3–5 s expressando a emoção (pode ser neutro)
    style_wav="Happy.wav",
    # informa o idioma ao modelo multilíngue
    language="pt-BR"
)

# 3) Salva o resultado
tts.save_wav(wav, "final_clone.wav")
print("✅ Gerado final_clone.wav")
