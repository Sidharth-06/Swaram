"""
Svaram Voice Engine — Inference Script
======================================
Tests the fully trained Parler-TTS Mini Expresso model with our custom
EmotionAdapter and LoRA weights.

Run:
    conda run -n parler_train python inference_svaram.py
"""

import os
import sys
import torch
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer, AutoFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration
from peft import PeftModel
from svaram.architecture import EmotionAdapter

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL = "parler-tts/parler-tts-mini-expresso"
CHECKPOINT = os.path.join(ROOT, "svaram_checkpoints", "step_epoch10")
OUTPUT_DIR = os.path.join(ROOT, "svaram_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def load_svaram_model():
    print(f"Loading base model ({BASE_MODEL})...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA weights from {CHECKPOINT}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT)

    print("Loading custom EmotionAdapter...")
    adapter_path = os.path.join(CHECKPOINT, "emotion_adapter.pt")
    emotion_adapter = EmotionAdapter(hidden_dim=768).to(device=device, dtype=dtype)
    
    if os.path.exists(adapter_path):
        emotion_adapter.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True))
        print(f"[OK] EmotionAdapter loaded successfully!")
    else:
        print(f"[WARNING] No emotion_adapter.pt found at {adapter_path}. Using untrained weights.")

    model.eval()
    emotion_adapter.eval()
    return model, tokenizer, feature_extractor, emotion_adapter

def generate_speech(model, tokenizer, feature_extractor, emotion_adapter, text, emotion="neutral", suffix=""):
    print(f"\n--- Generating: {emotion.upper()} ---")
    print(f"Text: '{text}'")

    description = "A female speaker with a clear, pleasant voice speaking at a natural pace."
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # 1. Run text encoder
        encoder_outputs = model.text_encoder(input_ids=input_ids)
        
        # 2. Inject emotion
        modified_hidden = emotion_adapter(
            encoder_outputs.last_hidden_state,
            emotion=emotion,
            text=text,
        )

        # 3. Project 768-dim → 1024-dim
        modified_hidden = model.enc_to_dec_proj(modified_hidden)
        encoder_outputs.last_hidden_state = modified_hidden

        # 4. Generate
        generation = model.generate(
            encoder_outputs=encoder_outputs,
            prompt_input_ids=prompt_input_ids,
            do_sample=True,
            temperature=1.0,
        )

    audio = generation[0, :].cpu().numpy().astype("float32")
    out_file = os.path.join(OUTPUT_DIR, f"svaram_{emotion}{suffix}.wav")
    sf.write(out_file, audio, feature_extractor.sampling_rate)
    print(f"[OK] Saved to {out_file}")

def main():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        return

    model, tokenizer, feature_extractor, emotion_adapter = load_svaram_model()

    # Test cases: Showing how to prompt Expresso for different intensities of non-speech sounds
    tests = [
        # 1. Neutral
        {
            "desc": "A female speaker with a clear, pleasant voice speaking at a natural pace.",
            "text": "This is the default voice of the Svaram engine. It should sound natural.",
            "name": "neutral"
        },
        # 2. Subtle Laugh
        {
            "desc": "A female speaker speaking with a clear voice. She speaks with laughter.",
            "text": "I can't believe we actually got this working! This is amazing!",
            "name": "happy_sublte_laugh"
        },
        # 3. Intense, Hard Laugh
        {
            "desc": "A female speaker giggling and bursting into cracking, intense laughter.",
            "text": "Ha ha ha! I can't believe this! That is so funny. Ha ha!",
            "name": "happy_intense_laugh"
        },
        # 4. Deep Sigh
        {
            "desc": "A female speaker sighing deeply and speaking with a sad tone.",
            "text": "I don't know if we can push this live. There are still some bugs.",
            "name": "sad_sighing"
        }
    ]

    for t in tests:
        # Pass the custom description instead of the default one
        print(f"\n--- Generating: {t['name'].upper()} ---")
        print(f"Desc: '{t['desc']}'")
        print(f"Text: '{t['text']}'")

        input_ids = tokenizer(t['desc'], return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(t['text'], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            encoder_outputs = model.text_encoder(input_ids=input_ids)
            
            # The emotion adapter isn't strictly necessary if the base prompt handles the emotion,
            # but it was trained alongside the LoRA weights, so we pass it through.
            modified_hidden = emotion_adapter(
                encoder_outputs.last_hidden_state,
                emotion="neutral", # Let the text description do the heavy lifting
                text=t['text'],
            )

            modified_hidden = model.enc_to_dec_proj(modified_hidden)
            encoder_outputs.last_hidden_state = modified_hidden

            generation = model.generate(
                encoder_outputs=encoder_outputs,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=1.0,
            )

        audio = generation[0, :].cpu().numpy().astype("float32")
        out_file = os.path.join(OUTPUT_DIR, f"svaram_final_{t['name']}.wav")
        sf.write(out_file, audio, feature_extractor.sampling_rate)
        print(f"[OK] Saved to {out_file}")

if __name__ == "__main__":
    main()
