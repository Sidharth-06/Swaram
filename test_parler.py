import torch
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor

def test_base_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_name = "parler-tts/parler-tts-mini-expresso"
    print(f"Loading {model_name}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    print("Model loaded successfully.")

    # Sample 1: Sighs and frustration
    prompt = "I don't know what to do anymore. It's just so frustrating!"
    description = "A female speaker sighs heavily before speaking with a slightly sad and expressive tone."

    print(f"\nGenerating Sample 1...")
    print(f"Prompt: {prompt}")
    print(f"Condition: {description}")
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation_dict = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        do_sample=True,
        temperature=1.0
    )

    audio = generation_dict[0, :].cpu().numpy().astype("float32")
    sf.write("test_parler_1.wav", audio, feature_extractor.sampling_rate)
    print("Saved -> test_parler_1.wav")

    # Sample 2: Laughter and excitement
    prompt = "Oh my goodness! That is the funniest thing I've ever heard!"
    description = "Thomas speaks with a laughing tone, laughing gleefully and speaking excitedly with an expressive voice."

    print(f"\nGenerating Sample 2...")
    print(f"Prompt: {prompt}")
    print(f"Condition: {description}")
    
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation_dict = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        do_sample=True,
        temperature=1.0
    )

    audio = generation_dict[0, :].cpu().numpy().astype("float32")
    sf.write("test_parler_2.wav", audio, feature_extractor.sampling_rate)
    print("Saved -> test_parler_2.wav")

if __name__ == "__main__":
    test_base_model()
