# vc.py
from TTS.api import TTS

# 1) Carrega um modelo que suporte VC zero-shot
#    YourTTS inclui VC além de TTS zero-shot
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# 2) Converte voice conversion
tts.voice_conversion_to_file(
    source_wav="models/xtts_v2/samples/pt_sample.wav",  # conteúdo/linguagem original
    target_wav="morty.wav",                              # timbre de destino (Morty)
    file_path="pt_to_morty.wav"                          # saída
)

print("✅ Gerado pt_to_morty.wav")
