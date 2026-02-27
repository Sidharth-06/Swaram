"""
Test script to figure out the exact prompting format for non-speech sounds
(laughs, sighs, pauses) in Parler-TTS Mini Expresso.
"""

import os
import torch
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    BASE_MODEL = "parler-tts/parler-tts-mini-expresso"
    
    print("Loading base Expresso model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    tests = [
        # Test 1: The standard Bark way (which we know fails for Expresso)
        {
            "desc": "A female speaker with a clear voice.",
            "text": "I can't believe this! [laughs] That is so funny.",
            "name": "01_bark_tags"
        },
        # Test 2: Putting the emotion explicitly in the description
        {
            "desc": "A female speaker laughing and chuckling while talking.",
            "text": "I can't believe this! That is so funny.",
            "name": "02_desc_laughing"
        },
        # Test 3: Using the known Expresso style metadata format
        {
            "desc": "A female speaker with a clear voice. She speaks with laughter.",
            "text": "I can't believe this! That is so funny.",
            "name": "03_desc_with_laughter"
        },
        # Test 4: Sighing test
        {
            "desc": "A female speaker sighing deeply.",
            "text": "I don't know if we can push this live. There are still some bugs.",
            "name": "04_desc_sighing"
        }
    ]

    os.makedirs("expresso_tests", exist_ok=True)

    print("\nRunning generation tests...")
    for t in tests:
        print(f"-> Testing: {t['name']}")
        input_ids = tokenizer(t['desc'], return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(t['text'], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=1.0,
            )
        
        audio = generation[0, :].cpu().numpy().astype("float32")
        sf.write(f"expresso_tests/{t['name']}.wav", audio, feature_extractor.sampling_rate)

    print("\nDone! Check the expresso_tests/ folder.")

if __name__ == "__main__":
    main()
