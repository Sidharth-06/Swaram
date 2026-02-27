"""
Pre-compute DAC audio codes for the Jenny TTS dataset.
Run this ONCE before training to eliminate the CPU bottleneck.

Output: parler_dataset_precomputed/
  - Each row has 'audio_codes' (tensor), 'transcription', etc.
  - No raw audio bytes → training loads only tensors.

Usage:
    conda run -n parler_train python precompute_codes.py
"""

import os, sys, io
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ["HF_HOME"] = os.path.join(ROOT, ".hf_cache")
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

from datasets import load_from_disk, Audio, Features, Value, Sequence
from transformers import AutoFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.modeling_parler_tts import build_delay_pattern_mask

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL     = "parler-tts/parler-tts-mini-expresso"
DATASET_PATH   = os.path.join(ROOT, "parler_dataset")
OUTPUT_PATH    = os.path.join(ROOT, "parler_dataset_precomputed")

MAX_AUDIO_LEN  = 400   # mel frames (~4s)
TARGET_SR      = 44100
NUM_CODEBOOKS  = 9
BOS_TOKEN_ID   = 1025
EOS_TOKEN_ID   = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Device: {device}")

    # Load model just for the audio encoder + feature extractor
    print("Loading base model for audio encoder...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    audio_encoder = model.audio_encoder.to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    del model.decoder, model.text_encoder  # free memory
    torch.cuda.empty_cache()
    print("[OK] Audio encoder loaded\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print(f"[OK] {len(dataset)} examples")

    max_samples = int(MAX_AUDIO_LEN * TARGET_SR / 86)  # ~4s of audio

    # Process all examples
    all_codes = []
    all_texts = []
    all_texts_norm = []
    skipped = 0

    print(f"\nEncoding {len(dataset)} audio files through DAC...")
    for i in tqdm(range(len(dataset)), desc="Encoding"):
        item = dataset[i]

        # ── Decode audio ──
        try:
            audio_data = item["audio"]
            raw_bytes = audio_data.get("bytes") or audio_data.get("array", None)
            if raw_bytes is None or not isinstance(raw_bytes, (bytes, bytearray)):
                skipped += 1
                continue

            arr, sr = sf.read(io.BytesIO(raw_bytes))
            arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            if sr != TARGET_SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)

            if len(arr) > max_samples:
                arr = arr[:max_samples]

        except Exception as e:
            print(f"  [SKIP] idx={i}: {e}")
            skipped += 1
            continue

        # ── Encode through DAC ──
        with torch.no_grad():
            mel_enc = feature_extractor([arr], sampling_rate=TARGET_SR, return_tensors="pt")
            mel_key = "input_features" if "input_features" in mel_enc else "input_values"
            input_values = mel_enc[mel_key].to(device=device, dtype=audio_encoder.dtype)

            audio_codes = audio_encoder.encode(input_values, return_dict=False)[0]
            if audio_codes.ndim == 4:
                audio_codes = audio_codes.squeeze(0)  # (frames,B,C,S) → (B,C,S)

            # Apply delay pattern mask (same as in training)
            bos_labels = torch.ones((1, NUM_CODEBOOKS, 1), dtype=torch.long, device=device) * BOS_TOKEN_ID
            audio_codes_with_bos = torch.cat([bos_labels, audio_codes], dim=-1)

            labels, delay_mask = build_delay_pattern_mask(
                audio_codes_with_bos,
                bos_token_id=BOS_TOKEN_ID,
                pad_token_id=EOS_TOKEN_ID,
                max_length=audio_codes_with_bos.shape[-1] + NUM_CODEBOOKS,
                num_codebooks=NUM_CODEBOOKS,
            )

            labels = torch.where(delay_mask == -1, EOS_TOKEN_ID, delay_mask)
            labels = labels[:, 1:]  # remove first BOS timestamp
            labels = labels.squeeze(0).transpose(0, 1)  # (1,C,S) → (S,C)

        all_codes.append(labels.cpu().numpy().tolist())
        all_texts.append(item.get("transcription", ""))
        all_texts_norm.append(item.get("transcription_normalised", ""))

    print(f"\n[OK] Encoded {len(all_codes)} / {len(dataset)} examples ({skipped} skipped)")

    # ── Save as a new HuggingFace dataset ──
    from datasets import Dataset as HFDataset

    new_ds = HFDataset.from_dict({
        "transcription":             all_texts,
        "transcription_normalised":  all_texts_norm,
        "labels":                    all_codes,
    })

    new_ds.save_to_disk(OUTPUT_PATH)
    print(f"\n[OK] Saved pre-computed dataset → {OUTPUT_PATH}")
    print(f"     {len(new_ds)} examples, columns: {new_ds.column_names}")

if __name__ == "__main__":
    main()
