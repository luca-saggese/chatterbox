import re
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

def main():
    # Read the text from the file
    try:
        with open('input.txt', 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("Error: The file 'input.txt' was not found.")
        return


    AUDIO_PROMPT_PATH = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac"
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    # Split the text into sentences using regular expressions
    #sentences = re.split(r'[!.?]+', text)
    sentences = re.split(r'\n\n', text)
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