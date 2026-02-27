"""
Svaram TTS — Fine-tuning Script
================================
Fine-tunes Parler-TTS Mini on the Jenny TTS dataset with our custom
EmotionAdapter and LoRA adapters.

Run:
    E:\\svaram\\tts\\miniconda3\\envs\\parler_train\\python.exe train.py

Hardware:
    - RTX 4060 8GB VRAM
    - Uses LoRA rank=16, 8-bit AdamW, gradient checkpointing
"""

import os
import sys
import io
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ["HF_HOME"] = os.path.join(ROOT, ".hf_cache")

# Force datasets to use soundfile instead of torchcodec (torchcodec not on Windows)
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

DATASET_PATH  = os.path.join(ROOT, "parler_dataset")
PRECOMPUTED_PATH = os.path.join(ROOT, "parler_dataset_precomputed")
CHECKPOINT_DIR = os.path.join(ROOT, "svaram_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Imports ────────────────────────────────────────────────────────────────
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio, Value
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    get_cosine_schedule_with_warmup,
)
from parler_tts import ParlerTTSForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from svaram.architecture import EmotionAdapter, EMOTION_LABELS

# ── Hyperparameters ────────────────────────────────────────────────────────
BASE_MODEL      = "parler-tts/parler-tts-mini-expresso"
BATCH_SIZE      = 2          # fits 8GB with gradient checkpointing
GRAD_ACCUM      = 8          # effective batch = 16
LR              = 1e-4
EPOCHS          = 10
WARMUP_STEPS    = 100
MAX_AUDIO_LEN   = 400        # mel frames (~4s at 24kHz, hop=256)
MAX_TEXT_LEN    = 128
LOG_EVERY       = 50
SAVE_EVERY      = 500
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


# ── Dataset Collator ───────────────────────────────────────────────────────

# ── Default Conditioning Description ──────────────────────────────────────
# Jenny TTS dataset doesn't have conditioning text. We synthesize one.
DEFAULT_DESCRIPTION = (
    "A female speaker with a clear, pleasant voice speaking at a natural pace. "
    "The recording is clean with minimal background noise."
)


from parler_tts import build_delay_pattern_mask
from torch.nn.utils.rnn import pad_sequence

class PrecomputedCollator:
    """
    FAST path: loads pre-computed DAC codes from disk.
    No audio decoding, no DAC encoding, no resampling.
    """
    def __init__(self, tokenizer, max_text_len):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch):
        texts        = [item["transcription"] for item in batch]
        descriptions = [DEFAULT_DESCRIPTION] * len(batch)

        desc_enc = self.tokenizer(
            descriptions, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt"
        )
        text_enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt"
        )

        # Labels are already pre-computed (seq_len, num_codebooks) stored as list-of-lists
        labels_list = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids":              desc_enc["input_ids"],
            "attention_mask":         desc_enc["attention_mask"],
            "prompt_input_ids":       text_enc["input_ids"],
            "prompt_attention_mask":   text_enc["attention_mask"],
            "labels":                 labels_padded,
        }


class JennyCollator:
    """
    Converts Jenny TTS dataset rows into Parler-TTS model inputs.
    """

    def __init__(self, tokenizer, feature_extractor, max_text_len, audio_encoder, device,
                 bos_token_id=1025, eos_token_id=1024, pad_token_id=1024, num_codebooks=9):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_text_len = max_text_len
        self.audio_encoder = audio_encoder
        self.device = device
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_codebooks = num_codebooks

    def __call__(self, batch):
        texts        = [item["transcription"] for item in batch]
        descriptions = [DEFAULT_DESCRIPTION]  * len(batch)

        audio_arrays = []
        sample_rates = []
        
        max_samples = int(MAX_AUDIO_LEN * 44100 / 86)
        
        for item in batch:
            audio_data = item["audio"]
            raw_bytes = audio_data.get("bytes") or audio_data.get("array", None)
            if raw_bytes is not None and isinstance(raw_bytes, (bytes, bytearray)):
                arr, sr = sf.read(io.BytesIO(raw_bytes))
                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)  # stereo → mono
                
                if sr != 44100:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=44100)
                    sr = 44100
                
                if len(arr) > max_samples:
                    arr = arr[:max_samples]
            else:
                raise ValueError(f"Cannot decode audio from keys: {list(audio_data.keys())}")
            audio_arrays.append(arr)
            sample_rates.append(sr)

        # Tokenize descriptions and text
        desc_enc = self.tokenizer(
            descriptions, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt"
        )
        text_enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt"
        )

        processed_labels = []
        bos_labels = torch.ones((1, self.num_codebooks, 1), dtype=torch.long, device=self.device) * self.bos_token_id
        
        with torch.no_grad():
            self.audio_encoder.eval()
            for arr in audio_arrays:
                # Extract features for EACH audio separately to get exact lengths
                mel_enc = self.feature_extractor([arr], sampling_rate=44100, return_tensors="pt")
                mel_key = "input_features" if "input_features" in mel_enc else "input_values"
                input_values = mel_enc[mel_key].to(device=self.device, dtype=self.audio_encoder.dtype)
                
                # audio_codes shape from encode: (frames, batch, codebooks, seq_len)
                # squeeze frames dim → (batch, codebooks, seq_len) to match bos_labels
                audio_codes = self.audio_encoder.encode(input_values, return_dict=False)[0]
                if audio_codes.ndim == 4:
                    audio_codes = audio_codes.squeeze(0)  # remove frames dim
                
                # apply delay pattern mask exactly as Parler-TTS does
                audio_codes_with_bos = torch.cat([bos_labels, audio_codes], dim=-1)
                
                labels, delay_pattern_mask = build_delay_pattern_mask(
                    audio_codes_with_bos,
                    bos_token_id=self.bos_token_id,
                    pad_token_id=self.eos_token_id,
                    max_length=audio_codes_with_bos.shape[-1] + self.num_codebooks,
                    num_codebooks=self.num_codebooks,
                )
                
                labels = torch.where(delay_pattern_mask == -1, self.eos_token_id, delay_pattern_mask)
                
                # remove first timestamp (full of BOS) -> matching ParlerTTS dataset postprocessing
                labels = labels[:, 1:] 
                
                # (1, num_codebooks, seq_len) -> (seq_len, num_codebooks)
                labels = labels.squeeze(0).transpose(0, 1)
                processed_labels.append(labels.cpu())

        # Pad the batch of labels with -100 so CrossEntropy ignores padding completely
        labels_padded = pad_sequence(processed_labels, batch_first=True, padding_value=-100)

        return {
            "input_ids":              desc_enc["input_ids"],
            "attention_mask":         desc_enc["attention_mask"],
            "prompt_input_ids":       text_enc["input_ids"],
            "prompt_attention_mask":  text_enc["attention_mask"],
            "labels":                 labels_padded,
        }



# ── LoRA Config ────────────────────────────────────────────────────────────

def apply_lora(model):
    """Apply LoRA adapters to attention layers (your trainable weights)."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Main Training Loop ─────────────────────────────────────────────────────

def train():
    print("\n-- Loading base model ----------------------------------")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
    )

    # Apply gradient checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[OK] Gradient checkpointing enabled")

    # Apply LoRA (your trainable adapter weights)
    model = apply_lora(model)

    # ── Attach custom EmotionAdapter ──
    emotion_adapter = EmotionAdapter(hidden_dim=768).to(device=device)  # FP32 — autocast handles FP16 in forward
    print(f"\n[OK] EmotionAdapter added ({sum(p.numel() for p in emotion_adapter.parameters()):,} params)")
    model = model.to(device)

    print("\n-- Loading dataset -------------------------------------")
    use_precomputed = os.path.exists(PRECOMPUTED_PATH)
    if use_precomputed:
        print("[FAST] Using pre-computed DAC codes!")
        dataset = load_from_disk(PRECOMPUTED_PATH)
        print(f"[OK] {len(dataset)} examples loaded (pre-computed)")
        print(f"   Columns: {dataset.column_names}")
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        val_ds   = split["test"]
        collator = PrecomputedCollator(tokenizer, MAX_TEXT_LEN)
    else:
        print("[SLOW] No pre-computed codes found. Run precompute_codes.py first for 2-3x speedup!")
        dataset = load_from_disk(DATASET_PATH)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        print(f"[OK] {len(dataset)} examples loaded (raw audio bytes)")
        print(f"   Columns: {dataset.column_names}")
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        val_ds   = split["test"]
        collator = JennyCollator(tokenizer, feature_extractor, MAX_TEXT_LEN, model.audio_encoder, device)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=False
    )

    # Optimizer: train LoRA + EmotionAdapter weights only
    trainable_params = (
        list(p for p in model.parameters() if p.requires_grad) +
        list(emotion_adapter.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-2)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    scaler = torch.amp.GradScaler(device='cuda')

    print(f"\n-- Training --------------------------------------------")
    print(f"   Epochs:    {EPOCHS}")
    print(f"   Batch:     {BATCH_SIZE} x {GRAD_ACCUM} grad accum = effective {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   Steps:     {total_steps}")
    print(f"   Optimizer: AdamW lr={LR}\n")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        emotion_adapter.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda'):
                # 1. Run text encoder to get base conditioning
                encoder_outputs = model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )

                # 2. Inject emotion via our custom EmotionAdapter (operates on 768-dim)
                modified_hidden = emotion_adapter(
                    encoder_outputs.last_hidden_state,
                    emotion="neutral",  # Jenny dataset has no emotion labels
                    text="",
                )

                # 3. Project 768-dim encoder states → 1024-dim decoder space
                #    (the model skips this when encoder_outputs is provided directly)
                modified_hidden = model.enc_to_dec_proj(modified_hidden)

                # 4. Apply attention mask to zero out padding positions
                if batch['attention_mask'] is not None:
                    modified_hidden = modified_hidden * batch['attention_mask'][..., None]

                encoder_outputs.last_hidden_state = modified_hidden

                # 5. Forward pass with modified encoder outputs (skip encoder re-run)
                outputs = model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=batch['attention_mask'],
                    prompt_input_ids=batch['prompt_input_ids'],
                    prompt_attention_mask=batch['prompt_attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step {global_step:5d} | loss {epoch_loss / (step+1):.4f} | lr {lr:.2e}")

                if global_step % SAVE_EVERY == 0:
                    _save(model, emotion_adapter, tokenizer, global_step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch}/{EPOCHS} — avg train loss: {avg_loss:.4f}\n")
        _save(model, emotion_adapter, tokenizer, f"epoch{epoch}")


def _save(model, emotion_adapter, tokenizer, tag):
    """Save LoRA weights + EmotionAdapter + tokenizer."""
    save_path = os.path.join(CHECKPOINT_DIR, f"step_{tag}")
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)          # saves LoRA adapter weights
    tokenizer.save_pretrained(save_path)
    torch.save(emotion_adapter.state_dict(), os.path.join(save_path, "emotion_adapter.pt"))
    print(f"  [SAVED] -> {save_path}")


if __name__ == "__main__":
    train()
