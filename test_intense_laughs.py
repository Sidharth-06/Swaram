"""
Test script to figure out how to force harder, less subtle laughs
in Parler-TTS Mini Expresso.
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

    # The text we want the model to laugh at
    joke_text = "I can't believe this! That is so funny."

    tests = [
        # Test 1: Combining emotion adverbs
        {
            "desc": "A female speaker speaking very happily with loud, uncontrollable laughter.",
            "text": joke_text,
            "name": "01_loud_uncontrollable_laughter"
        },
        # Test 2: Putting the laugh action at the beginning AND end of the prompt
        {
            "desc": "Laughing hysterically, a female speaker with a clear voice speaks with heavy laughter.",
            "text": joke_text,
            "name": "02_hysterically_heavy"
        },
        # Test 3: Using extreme adjectives
        {
            "desc": "A female speaker giggling and bursting into cracking, intense laughter.",
            "text": joke_text,
            "name": "03_bursting_intense"
        },
        # Test 4: Using the text itself to guide the laughter (e.g., adding "Hahaha" to the text, but asking it to speak those words while laughing)
        {
            "desc": "A female speaker speaking with loud laughter and joy.",
            "text": "Ha ha ha! I can't believe this! That is so funny. Ha ha!",
            "name": "04_haha_text_plus_laughter_desc"
        }
    ]

    os.makedirs("expresso_intense_laughs", exist_ok=True)

    print("\nRunning intense laughter generation tests...")
    for t in tests:
        print(f"-> Testing: {t['name']}")
        input_ids = tokenizer(t['desc'], return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(t['text'], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=1.0,  # Try keeping temperature at 1.0; increasing it too much degrades speech
            )
        
        audio = generation[0, :].cpu().numpy().astype("float32")
        sf.write(f"expresso_intense_laughs/{t['name']}.wav", audio, feature_extractor.sampling_rate)

    print("\nDone! Check the expresso_intense_laughs/ folder.")

if __name__ == "__main__":
    main()
