#!/usr/bin/env python3
"""
Improved Long Audio Generator - Generates long audio without problematic chunking
Based on the multilingual_app.py approach but optimized for longer texts
"""

import torch
import numpy as np
import soundfile as sf
import os
import re
from typing import Optional, Tuple, List
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

class LongAudioGenerator:
    """Generator for long audio content without chunking artifacts"""
    
    def __init__(self, device: str = None):
        """Initialize the generator with the multilingual TTS model"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the ChatterboxMultilingualTTS model"""
        print(f"🚀 Loading ChatterboxMultilingualTTS on {self.device}...")
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(self.device)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible results"""
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def smart_split_text(self, text: str, max_length: int = 500) -> List[str]:
        """
        Intelligently split text into chunks without breaking sentences
        This is used only when text is extremely long to prevent memory issues
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max_length, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_length:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If still too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split by sentences
                sentences = re.split(r'([.!?]+\s*)', chunk)
                current_sentence_chunk = ""
                
                i = 0
                while i < len(sentences):
                    sentence = sentences[i].strip()
                    if not sentence:
                        i += 1
                        continue
                    
                    # Add punctuation if it exists
                    if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
                        sentence += sentences[i + 1]
                        i += 2
                    else:
                        i += 1
                    
                    # Check if adding this sentence would exceed max_length
                    if current_sentence_chunk and len(current_sentence_chunk) + len(sentence) + 1 > max_length:
                        if current_sentence_chunk.strip():
                            final_chunks.append(current_sentence_chunk.strip())
                        current_sentence_chunk = sentence
                    else:
                        if current_sentence_chunk:
                            current_sentence_chunk += " " + sentence
                        else:
                            current_sentence_chunk = sentence
                
                # Add the last sentence chunk
                if current_sentence_chunk.strip():
                    final_chunks.append(current_sentence_chunk.strip())
        
        return final_chunks
    
    def generate_long_audio(
        self,
        text: str,
        language_id: str = "it",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int = 0,
        max_chunk_length: int = 100,  # Increased from 500
        add_pauses: bool = False,
        pause_duration: float = 0.5
    ) -> Tuple[int, np.ndarray]:
        """
        Generate long audio from text with minimal chunking
        
        Args:
            text: Input text
            language_id: Language code (e.g., 'it', 'en', 'fr')
            audio_prompt_path: Optional reference audio file
            exaggeration: Speech expressiveness (0.25-2.0)
            temperature: Randomness (0.05-5.0)
            cfg_weight: Generation guidance (0.2-1.0)
            seed: Random seed (0 for random)
            max_chunk_length: Maximum characters per chunk (only for very long texts)
            add_pauses: Whether to add pauses between chunks
            pause_duration: Pause duration in seconds
        """
        if not self.model:
            raise RuntimeError("TTS model is not loaded")
        
        if seed != 0:
            self.set_seed(seed)
        
        print(f"🎤 Generating audio for {len(text)} characters in {language_id}")
        
        # For most texts, try to generate without chunking
        if len(text) <= max_chunk_length:
            print("📝 Generating audio without chunking...")
            try:
                wav = self._generate_single_chunk(
                    text, language_id, audio_prompt_path, 
                    exaggeration, temperature, cfg_weight
                )
                return (self.model.sr, wav.squeeze(0).numpy())
            except Exception as e:
                print(f"⚠️ Single chunk generation failed: {e}")
                print("🔄 Falling back to smart chunking...")
        
        # Fallback to smart chunking for very long texts or if single generation fails
        chunks = self.smart_split_text(text, max_chunk_length)
        print(f"📄 Split into {len(chunks)} chunks")
        
        audio_segments = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"🎵 Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            try:
                wav = self._generate_single_chunk(
                    chunk, language_id, audio_prompt_path,
                    exaggeration, temperature, cfg_weight
                )
                
                # DEBUG: Save individual chunk for inspection
                chunk_audio = wav.squeeze(0).numpy()
                debug_filename = f"debug_chunk_{i:03d}_of_{len(chunks):03d}.wav"
                sf.write(debug_filename, chunk_audio, self.model.sr)
                print(f"🔍 DEBUG: Saved chunk {i} to {debug_filename} ({len(chunk_audio)/self.model.sr:.2f}s)")
                
                audio_segments.append(chunk_audio)
                
                # Add pause between chunks (except for the last one)
                if add_pauses and i < len(chunks):
                    pause_samples = int(pause_duration * self.model.sr)
                    pause = np.zeros(pause_samples, dtype=np.float32)
                    audio_segments.append(pause)
                    
            except Exception as e:
                print(f"❌ Error generating chunk {i}: {e}")
                # Add silence instead of failing completely
                fallback_duration = 1.0  # 1 second of silence
                fallback_samples = int(fallback_duration * self.model.sr)
                fallback_audio = np.zeros(fallback_samples, dtype=np.float32)
                audio_segments.append(fallback_audio)
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            print(f"✅ Generated {len(combined_audio)/self.model.sr:.1f} seconds of audio")
            return (self.model.sr, combined_audio)
        else:
            raise RuntimeError("Failed to generate any audio")
    
    def _generate_single_chunk(
        self, 
        text: str, 
        language_id: str, 
        audio_prompt_path: Optional[str],
        exaggeration: float,
        temperature: float,
        cfg_weight: float
    ) -> torch.Tensor:
        """Generate audio for a single text chunk"""
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path
        
        # Use the same approach as multilingual_app.py
        wav = self.model.generate(
            text,
            language_id=language_id,
            # Enable stabilized inference by default for long text
            stable_inference=True,
            stable_position_mode='cyclic',
            stable_max_position_embedding=256,
            stable_reset_position_every=None,
            stable_stabilize_kv_cache=False,
            stable_kv_reset_interval=0,
            **generate_kwargs
        )
        
        return wav
    
    def save_audio(
        self, 
        audio_data: Tuple[int, np.ndarray], 
        output_path: str
    ) -> str:
        """Save generated audio to file"""
        sample_rate, audio_array = audio_data
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio_array, sample_rate)
        print(f"💾 Audio saved to: {output_path}")
        
        # Return file info
        duration = len(audio_array) / sample_rate
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        return f"✅ Saved: {output_path}\n⏱️ Duration: {duration:.1f}s\n📁 Size: {file_size:.1f} MB"


def test_long_audio_generation():
    """Test the long audio generator with Italian text"""
    
    # Initialize generator
    generator = LongAudioGenerator()
    
    # Test text with accented characters
    test_text = """
    Ciao! Benvenuti a questo test di generazione audio lunga in italiano. 
    
    Questo è un testo molto più lungo che contiene diverse frasi con lettere accentate. 
    Per esempio: è, così, più, perché, università, città. 
    
    Il nostro sistema dovrebbe essere in grado di gestire testi lunghi senza problemi di chunking 
    che causano ripetizioni o artefatti audio. 
    
    Vediamo se funziona correttamente con questo approccio migliorato che utilizza 
    il metodo diretto del modello multilingual invece del chunking problematico.
    
    Questo paragrafo è ancora più lungo per testare la capacità del sistema di gestire 
    contenuti estesi. Dovrebbe mantenere la fluidità e la naturalezza della voce 
    senza interruzioni o ripetizioni fastidiose.
    """
    
    try:
        # Generate audio
        print("🎬 Starting long audio generation test...")
        audio_data = generator.generate_long_audio(
            text=test_text,
            language_id="it",
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            max_chunk_length=800,  # Larger chunks for better continuity
            add_pauses=True,
            pause_duration=0.3
        )
        
        # Save audio
        output_file = "test_long_audio_italian.wav"
        result = generator.save_audio(audio_data, output_file)
        print(result)
        
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_long_audio_generation()
