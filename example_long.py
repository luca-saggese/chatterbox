import re
import torchaudio as ta
import torch
import urllib.request
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

def download_audio_if_needed(audio_path: str) -> str:
    """Download audio file from URL if needed, otherwise return the path as-is."""
    if audio_path.startswith(('http://', 'https://')):
        # Create a temp directory if it doesn't exist
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract filename from URL
        filename = audio_path.split('/')[-1]
        local_path = temp_dir / filename
        
        # Download if not already cached
        if not local_path.exists():
            print(f"Downloading audio from {audio_path}...")
            urllib.request.urlretrieve(audio_path, local_path)
            print(f"Downloaded to {local_path}")
        else:
            print(f"Using cached audio: {local_path}")
        
        return str(local_path)
    return audio_path

def main():
    # Read the text from the file
    try:
        with open('input.txt', 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("Error: The file 'input.txt' was not found.")
        return

    # AUDIO_PROMPT_PATH = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac"
    # # Download the audio file if it's a URL
    # AUDIO_PROMPT_PATH = download_audio_if_needed(AUDIO_PROMPT_PATH)
    AUDIO_PROMPT_PATH = "pannofino.mp3"
    
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    
    # The model handles text splitting automatically!
    # Use split_mode="adaptive" for books - groups sentences to ~800 chars
    # This maintains narrative context while optimizing generation
    wav = model.generate(
        text, 
        cfg_weight=0.2,
        exaggeration=1.5,
        language_id="it", 
        audio_prompt_path=AUDIO_PROMPT_PATH,
        auto_split=True,  # Enable automatic splitting
        split_mode="adaptive",  # Adaptive chunking by character count (best for books)
        target_chars=800,  # Target ~800 chars per chunk (3-5 sentences)
        max_new_tokens=2000  # Increase if needed for longer chunks
    )
    
    # Save the combined audio
    ta.save("out.wav", wav, model.sr)
    print("Audio saved to out.wav")

if __name__ == "__main__":
    main()