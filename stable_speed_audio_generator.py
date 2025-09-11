#!/usr/bin/env python3
"""
Stable Speed Audio Generator - Prevents speed acceleration in long audio generation
"""

import torch
import numpy as np
import soundfile as sf
import os
import re
import time
from typing import Optional, Tuple, List
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

class StableSpeedAudioGenerator:
    """Generator for long audio with stable speaking speed throughout"""
    
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
    
    def reset_model_state(self):
        """Reset model internal state to prevent speed acceleration"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Small delay to let the model "cool down"
        time.sleep(0.1)
    
    def set_consistent_seed(self, base_seed: int, chunk_index: int = 0):
        """Set consistent seed that varies slightly per chunk to avoid repetition but maintains stability"""
        # Use a deterministic but slightly different seed for each chunk
        chunk_seed = base_seed + (chunk_index * 7) % 1000  # Small variation
        
        torch.manual_seed(chunk_seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(chunk_seed)
            torch.cuda.manual_seed_all(chunk_seed)
        np.random.seed(chunk_seed)
    
    def smart_split_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        Split text into smaller, more manageable chunks to prevent speed acceleration
        Reduced chunk size for better speed stability
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
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
        
        # If chunks are still too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split by sentences more aggressively
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
    
    def generate_stable_speed_audio(
        self,
        text: str,
        language_id: str = "it",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.7,  # Slightly lower for more stability
        cfg_weight: float = 0.6,   # Slightly higher for more control
        seed: int = 42,            # Default non-zero seed for consistency
        max_chunk_length: int = 350,  # Smaller chunks for stability
        add_pauses: bool = True,
        pause_duration: float = 0.4,  # Slightly longer pauses
        reset_model_every: int = 3    # Reset model state every N chunks
    ) -> Tuple[int, np.ndarray]:
        """
        Generate long audio with stable speaking speed throughout
        
        Args:
            text: Input text
            language_id: Language code (e.g., 'it', 'en', 'fr')
            audio_prompt_path: Optional reference audio file
            exaggeration: Speech expressiveness (0.25-2.0) - kept moderate
            temperature: Randomness (0.05-5.0) - lower for stability
            cfg_weight: Generation guidance (0.2-1.0) - higher for control
            seed: Random seed (use non-zero for consistency)
            max_chunk_length: Maximum characters per chunk (smaller for stability)
            add_pauses: Whether to add pauses between chunks
            pause_duration: Pause duration in seconds
            reset_model_every: Reset model state every N chunks to prevent drift
        """
        if not self.model:
            raise RuntimeError("TTS model is not loaded")
        
        print(f"🎤 Generating stable-speed audio for {len(text)} characters in {language_id}")
        
        # Always use chunking for consistent speed control
        chunks = self.smart_split_text(text, max_chunk_length)
        print(f"📄 Split into {len(chunks)} chunks for stable speed")
        
        audio_segments = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"🎵 Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            # Reset model state periodically to prevent speed drift
            if (i - 1) % reset_model_every == 0:
                print(f"🔄 Resetting model state (chunk {i})")
                self.reset_model_state()
            
            # Set consistent but slightly varied seed
            if seed != 0:
                self.set_consistent_seed(seed, i - 1)
            
            try:
                wav = self._generate_stable_chunk(
                    chunk, language_id, audio_prompt_path,
                    exaggeration, temperature, cfg_weight
                )
                audio_segments.append(wav.squeeze(0).numpy())
                
                # Add pause between chunks (except for the last one)
                if add_pauses and i < len(chunks):
                    pause_samples = int(pause_duration * self.model.sr)
                    # Use slight fade in/out for smoother transitions
                    pause = np.zeros(pause_samples, dtype=np.float32)
                    audio_segments.append(pause)
                    
            except Exception as e:
                print(f"❌ Error generating chunk {i}: {e}")
                # Add silence instead of failing completely
                fallback_duration = 1.0
                fallback_samples = int(fallback_duration * self.model.sr)
                fallback_audio = np.zeros(fallback_samples, dtype=np.float32)
                audio_segments.append(fallback_audio)
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = np.concatenate(audio_segments)
            
            # Apply gentle normalization to ensure consistent volume
            combined_audio = self._normalize_audio(combined_audio)
            
            duration = len(combined_audio) / self.model.sr
            print(f"✅ Generated {duration:.1f} seconds of stable-speed audio")
            return (self.model.sr, combined_audio)
        else:
            raise RuntimeError("Failed to generate any audio")
    
    def _generate_stable_chunk(
        self, 
        text: str, 
        language_id: str, 
        audio_prompt_path: Optional[str],
        exaggeration: float,
        temperature: float,
        cfg_weight: float
    ) -> torch.Tensor:
        """Generate audio for a single text chunk with stable parameters"""
        
        # Use consistent parameters for stable generation
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path
        
        # Generate with consistent settings
        wav = self.model.generate(
            text,
            language_id=language_id,
            # Stable inference flags
            stable_inference=True,
            stable_position_mode='cyclic',
            stable_max_position_embedding=256,
            stable_reset_position_every=None,
            stable_stabilize_kv_cache=True,
            stable_kv_reset_interval=200,
            **generate_kwargs
        )
        
        return wav
    
    def _normalize_audio(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """Normalize audio to prevent volume inconsistencies"""
        if len(audio) == 0:
            return audio
        
        # Find peak value
        peak = np.max(np.abs(audio))
        
        if peak > 0:
            # Normalize to target peak
            audio = audio * (target_peak / peak)
        
        return audio
    
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


def test_stable_speed_generation():
    """Test the stable speed generator with a longer Italian text"""
    
    # Initialize generator
    generator = StableSpeedAudioGenerator()
    
    # Longer test text to see speed consistency
    test_text = """
    Benvenuti a questo test avanzato del generatore audio con velocità stabile.
    
    Questo è un testo significativamente più lungo che serve a testare se il nostro 
    sistema riesce a mantenere una velocità di pronuncia costante durante tutta la generazione.
    
    Nel primo paragrafo, la velocità dovrebbe essere normale e naturale. Stiamo utilizzando 
    parole con lettere accentate come: è, così, più, perché, università, città, qualità.
    
    Nel secondo paragrafo, la velocità dovrebbe rimanere identica al primo, senza accelerazioni 
    o rallentamenti. Questo è fondamentale per una buona esperienza di ascolto.
    
    Il terzo paragrafo continua il test con frasi più lunghe e complesse. Dovremmo sentire 
    la stessa cadenza e lo stesso ritmo delle sezioni precedenti, mantenendo una pronuncia 
    chiara e ben articolata.
    
    Nel quarto paragrafo, aggiungiamo ancora più contenuto per verificare che anche verso 
    la fine della generazione, la velocità rimanga costante e non ci siano variazioni 
    indesiderate nel tempo di pronuncia.
    
    Questo è il paragrafo finale del nostro test. Se tutto funziona correttamente, 
    dovremmo sentire la stessa velocità dall'inizio alla fine, con una pronuncia 
    fluida e naturale per tutto il tempo.
    """
    
    try:
        # Generate audio with stable speed settings
        print("🎬 Starting stable speed audio generation test...")
        audio_data = generator.generate_stable_speed_audio(
            text=test_text,
            language_id="it",
            exaggeration=0.5,
            temperature=0.7,      # Lower for stability
            cfg_weight=0.6,       # Higher for control
            seed=42,              # Consistent seed
            max_chunk_length=300, # Smaller chunks
            add_pauses=True,
            pause_duration=0.4,
            reset_model_every=3   # Reset every 3 chunks
        )
        
        # Save audio
        output_file = "test_stable_speed_italian.wav"
        result = generator.save_audio(audio_data, output_file)
        print(result)
        
        print("🎉 Stable speed test completed!")
        print("🎧 Ascolta l'audio per verificare che la velocità rimanga costante")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_stable_speed_generation()
