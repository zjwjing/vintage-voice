#!/usr/bin/env python3
"""A/B: CosyVoice2-cajun-ep2 (current) vs CosyVoice2-cajun-prairie-ep1 (new).
CPU-only so it won't fight the llamas. Lines run through the Louisiana lexicon."""
import sys, os, subprocess
BASE = "/home/scott/vintage-voice"
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo")
sys.path.insert(0, f"{BASE}/models/cosyvoice-repo/third_party/Matcha-TTS")
sys.path.insert(0, f"{BASE}/scripts/cajun8h")
import torch, torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cajun_lexicon import respell
assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"

REF_AUDIO = "/home/scott/vintage-voice-samples-50ep/sophia_ref.wav"
REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
            "trapped and tagged approximately 15.5 of these majestic little creatures.")
LINES = [
    "Mais comment ça va, cher? On va au fais do-do à soir, tout le monde va danser!",
    "Laissez les bons temps rouler! On fait un bon gombo, et après on va au fais do-do.",
]
MODELS = {
    "ep2":          f"{BASE}/models/CosyVoice2-cajun-ep2",
    "prairie-ep1":  f"{BASE}/models/CosyVoice2-cajun-prairie-ep1",
}
OUT = f"{BASE}/data/output/prairie_ab"

def loud(p):
    t = p + ".l.wav"
    if subprocess.run(["ffmpeg","-y","-loglevel","error","-i",p,
                       "-af","loudnorm=I=-12:LRA=7:TP=-1.0",t]).returncode == 0:
        os.replace(t, p)

def main():
    for tag, mdir in MODELS.items():
        d = f"{OUT}/{tag}"; os.makedirs(d, exist_ok=True)
        print(f"\n===== {tag} =====", flush=True)
        m = CosyVoice2(mdir, load_jit=False, load_trt=False, fp16=False)
        for n, line in enumerate(LINES, 1):
            spoken = respell(line)
            for i, o in enumerate(m.inference_zero_shot(spoken, REF_TEXT, REF_AUDIO, stream=False)):
                p = f"{d}/line_{n}.wav" if i == 0 else f"{d}/line_{n}_{i}.wav"
                torchaudio.save(p, o["tts_speech"], m.sample_rate)
                loud(p)
                print(f"  {p}", flush=True)
        del m
        import gc; gc.collect()
    print("\nAB_DONE", flush=True)

if __name__ == "__main__":
    main()
