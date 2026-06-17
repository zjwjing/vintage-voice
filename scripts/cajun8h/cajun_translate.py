#!/usr/bin/env python3
"""
English -> Cajun French -> Sophia's voice.  The full tool.

Pipeline:
  1. translate_en_to_cajun(text)  -> local qwen-7b (:8081), Cajun-biased prompt
  2. respell(cajun)               -> Louisiana pronunciation lexicon
  3. CosyVoice2-cajun-ep2          -> Sophia's Cajun voice + loudnorm

Usage (translate + speak):
  CUDA_VISIBLE_DEVICES="" /home/scott/vintage-voice/venv-cosy/bin/python \
      /home/scott/cajun_translate.py "Come eat, the gumbo is ready!" gumbo

Usage (translate only, no audio):
  .../python /home/scott/cajun_translate.py --text-only "Watch out for the mosquitoes"

Set CUDA_VISIBLE_DEVICES=0 to use the GPU (stop the llamas first).
"""
import sys, os, json, subprocess
import urllib.request

BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/scripts/cajun8h")
from cajun_lexicon import respell  # noqa: E402

LLM_URL = "http://127.0.0.1:8081/v1/chat/completions"
LLM_KEY = "local"

# Bias HARD toward Louisiana Cajun French, not Parisian. Few-shot anchors the
# register so the model reaches for maringouin/asteur/icitte instead of the
# textbook word, and outputs ONLY the sentence (no notes to leak into TTS).
SYS_PROMPT = (
    "You are a Louisiana Cajun French translator from Acadiana. Translate the "
    "English into AUTHENTIC spoken Louisiana Cajun French (Cadien), NOT standard "
    "modern Parisian French.\n"
    "HISTORY (obey it): Cajun French is essentially 17th-18th-century colonial "
    "French (circa 1750s Acadian/provincial French) with Native American, Spanish, "
    "and other influences layered in — NOT modern Parisian. Prefer older/archaic "
    "and provincial French forms and vocabulary over modern coinages. When NO "
    "French/Cajun word exists for a thing, borrow the English word (lightly "
    "Frenchified is fine) — Cajuns do exactly this; do NOT invent a Parisian "
    "neologism.\n"
    "Prefer Cajun words and idiom: maringouin (not moustique), asteur (now), "
    "icitte (here), sha (never 'chère'), mais, ouais, lâche pas la patate, "
    "gardez-moi ça, couillon, bétaille, char (car), nonc, tante, quéque chose. "
    "Keep contractions casual and spoken (j'te dis, not 'je vais te dire'), like "
    "family talking around the table. Output ONLY the Cajun French sentence — no "
    "quotes, no English glosses, no notes, no explanation."
)
FEWSHOT = [
    ("Hello my friend, how are you doing?", "Mais bonjour sha, comment ça va, toi ?"),
    ("Watch out for the mosquitoes tonight.", "Fais attention aux maringouins à soir."),
    ("Come eat, the food is ready now.", "Viens manger, le manger est prêt asteur."),
]


def translate_en_to_cajun(text: str, temperature: float = 0.4) -> str:
    """English -> Louisiana Cajun French via the local 7B. Returns one line."""
    msgs = [{"role": "system", "content": SYS_PROMPT}]
    for en, cj in FEWSHOT:
        msgs.append({"role": "user", "content": en})
        msgs.append({"role": "assistant", "content": cj})
    msgs.append({"role": "user", "content": text})
    payload = json.dumps({
        "model": "qwen", "messages": msgs,
        "max_tokens": 160, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        LLM_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    out = data["choices"][0]["message"]["content"].strip()
    # strip stray wrapping quotes / leading label if the model adds one
    out = out.strip().strip('"').strip()
    return out.splitlines()[0].strip() if out else out


def speak(cajun_text: str, name: str = "cajun"):
    """respell + CosyVoice2-cajun-ep2 + loudnorm. Writes a wav, returns path."""
    sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
    sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
    import torch, torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2

    MODEL = f"{BASE}/models/CosyVoice2-cajun-ep2"
    REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
    REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
                "trapped and tagged approximately 15.5 of these majestic little creatures.")
    OUT = f"{BASE}/data/output/cajun_translate"
    os.makedirs(OUT, exist_ok=True)

    spoken = respell(cajun_text)
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    print(f"loading epoch-2 Cajun model on {'GPU' if on_gpu else 'CPU'}...", flush=True)
    model = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=on_gpu)
    path = f"{OUT}/{name}.wav"
    for i, o in enumerate(model.inference_zero_shot(spoken, REF_TEXT, REF_AUDIO, stream=False)):
        path = f"{OUT}/{name}.wav" if i == 0 else f"{OUT}/{name}_{i}.wav"
        torchaudio.save(path, o["tts_speech"], model.sample_rate)
        tmp = path + ".loud.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                           "-af", "loudnorm=I=-12:LRA=7:TP=-1.0", tmp]).returncode == 0:
            os.replace(tmp, path)
        print("WROTE", path, flush=True)
    return path


def main():
    args = sys.argv[1:]
    text_only = False
    if args and args[0] == "--text-only":
        text_only = True
        args = args[1:]
    if not args:
        print('usage: cajun_translate.py [--text-only] "<english>" [out_basename]')
        sys.exit(1)
    english = args[0]
    name = args[1] if len(args) > 1 else "cajun"

    print(f"EN : {english}", flush=True)
    cajun = translate_en_to_cajun(english)
    print(f"CJ : {cajun}", flush=True)
    print(f"SAY: {respell(cajun)}", flush=True)
    if not text_only:
        speak(cajun, name)


if __name__ == "__main__":
    main()
