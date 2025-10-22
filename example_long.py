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


    AUDIO_PROMPT_PATH = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac"
    # Download the audio file if it's a URL
    AUDIO_PROMPT_PATH = download_audio_if_needed(AUDIO_PROMPT_PATH)
    
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    # Split the text into sentences using regular expressions
    #sentences = re.split(r'[!.?]+', text)
    sentences = re.split(r'[\.\!\?\n]', text)
    filtered_sentences = [s.strip() for s in sentences if s.strip()]
    i=0
    combined_audio=None
    # Call the speak function for each sentence
    for sentence in filtered_sentences:

        wav = model.generate(sentence, language_id="it", audio_prompt_path=AUDIO_PROMPT_PATH)
        filename = "test-" + str(i) + ".wav"

        #if i!=0:
        if combined_audio is None:
            combined_audio= wav # Initialize with the first audio tensor
        else:
            combined_audio = torch.cat((combined_audio,wav),dim=1)

        ta.save(filename, wav, model.sr)
        i+=1

    ta.save("out.wav", combined_audio, model.sr)

if __name__ == "__main__":
    main()