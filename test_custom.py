"""
Test the fixed Svaram custom model architecture:
  - Base Parler-TTS Mini Expresso
  - EmotionAdapter (768-dim, matching T5 encoder)
  - LoRA adapters on decoder attention layers

This script tests the architecture with freshly initialized weights
since the old checkpoints used the wrong hidden_dim (1024 vs 768).
To test with actual trained weights, retrain with the fixed train.py first.
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
from peft import LoraConfig, get_peft_model, TaskType
from svaram.architecture import EmotionAdapter, EMOTION_LABELS

def test_architecture():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    BASE_MODEL = "parler-tts/parler-tts-mini-expresso"

    print(f"Loading base model {BASE_MODEL}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    # Apply LoRA (freshly initialized — just testing architecture works)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Create EmotionAdapter with correct dim (768 = T5 encoder d_model)
    emotion_adapter = EmotionAdapter(hidden_dim=768).to(device, dtype=dtype)
    print(f"EmotionAdapter: {sum(p.numel() for p in emotion_adapter.parameters()):,} params")

    model.eval()
    emotion_adapter.eval()
    print("Architecture ready.\n")

    # Test a single generation to verify the full pipeline works
    prompt = "I really hope this custom model sounds amazing."
    description = "A female speaker with a clear, pleasant voice speaking at a natural pace."
    emotion = "neutral"

    print(f"--- Generating with emotion: '{emotion}' ---")
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # 1. Run text encoder (outputs 768-dim hidden states)
        encoder_outputs = model.text_encoder(input_ids=input_ids)
        
        # 2. Inject emotion via EmotionAdapter (still 768-dim)
        modified_hidden = emotion_adapter(
            encoder_outputs.last_hidden_state,
            emotion=emotion,
            text=prompt,
        )

        # 3. Project 768-dim → 1024-dim (the model skips this when encoder_outputs is provided)
        modified_hidden = model.enc_to_dec_proj(modified_hidden)

        # 4. Store the projected hidden states back
        encoder_outputs.last_hidden_state = modified_hidden

        # 5. Generate with modified hidden states
        generation = model.generate(
            encoder_outputs=encoder_outputs,
            prompt_input_ids=prompt_input_ids,
            do_sample=True,
            temperature=1.0,
        )

    audio = generation[0, :].cpu().numpy().astype("float32")
    out_file = f"test_emotion_{emotion}.wav"
    sf.write(out_file, audio, feature_extractor.sampling_rate)
    print(f"  Saved -> {out_file}")
    print("\nArchitecture test PASSED! The full pipeline works end-to-end.")

if __name__ == "__main__":
    test_architecture()
