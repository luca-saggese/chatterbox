#!/usr/bin/env python3
"""
Optimized Long Text Audio Generator
Specializzato per testi molto lunghi con chunking intelligente e senza pause artificiali
"""

import torch
import numpy as np
import soundfile as sf
import os
import re
from typing import Optional, Tuple, List
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

class OptimizedLongTextGenerator:
    """Generator ottimizzato per testi molto lunghi"""
    
    def __init__(self, device: str = None):
        """Initialize the generator"""
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
    
    def optimal_text_chunking(self, text: str, max_chars: int = 1200) -> List[str]:
        """
        Chunking ottimizzato che mantiene il flow naturale del discorso
        """
        if len(text) <= max_chars:
            return [text.strip()]
        
        chunks = []
        
        # Prima dividi per paragrafi
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Se il paragrafo da solo supera la lunghezza massima, dividilo per frasi
            if len(paragraph) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Dividi il paragrafo lungo in frasi
                sentences = self._split_into_sentences(paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 <= max_chars:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
            
            # Se aggiungere questo paragrafo supererebbe la lunghezza massima
            elif current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            
            # Altrimenti aggiungilo al chunk corrente
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Aggiungi l'ultimo chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Dividi il testo in frasi mantenendo la punteggiatura"""
        # Pattern per dividere le frasi
        sentence_pattern = r'([.!?]+)\s+'
        parts = re.split(sentence_pattern, text)
        
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(r'[.!?]+', parts[i + 1]):
                # Combina la frase con la sua punteggiatura
                sentence = parts[i] + parts[i + 1]
                sentences.append(sentence.strip())
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i].strip())
                i += 1
        
        return [s for s in sentences if s.strip()]
    
    def generate_long_text_audio(
        self,
        text: str,
        language_id: str = "it",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int = 0,
        max_chunk_chars: int = 1200,
        natural_pauses: bool = True,
        pause_duration: float = 0.1
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio for very long texts with optimized chunking
        
        Args:
            text: Input text (può essere molto lungo)
            language_id: Language code
            audio_prompt_path: Optional reference audio
            exaggeration: Speech expressiveness (0.25-2.0)
            temperature: Randomness (0.05-5.0) 
            cfg_weight: Generation guidance (0.2-1.0)
            seed: Random seed
            max_chunk_chars: Maximum characters per chunk
            natural_pauses: Add very short natural pauses between chunks
            pause_duration: Duration of natural pauses in seconds
        """
        if not self.model:
            raise RuntimeError("TTS model is not loaded")
        
        if seed != 0:
            self._set_seed(seed)
        
        # Clean and prepare text
        clean_text = self._clean_text(text)
        print(f"🎤 Generating audio for {len(clean_text)} characters in {language_id}")
        
        # Try single generation first for shorter texts
        if len(clean_text) <= max_chunk_chars:
            print("📝 Generating as single chunk...")
            try:
                wav = self._generate_chunk(
                    clean_text, language_id, audio_prompt_path,
                    exaggeration, temperature, cfg_weight
                )
                return (self.model.sr, wav.squeeze(0).numpy())
            except Exception as e:
                print(f"⚠️ Single chunk failed: {e}")
        
        # Use optimized chunking for long texts
        chunks = self.optimal_text_chunking(clean_text, max_chunk_chars)
        print(f"📄 Optimally split into {len(chunks)} chunks")
        
        audio_segments = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"🎵 Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            try:
                # Reset seed for each chunk to maintain consistency
                if seed != 0:
                    self._set_seed(seed + i)
                
                wav = self._generate_chunk(
                    chunk, language_id, audio_prompt_path,
                    exaggeration, temperature, cfg_weight
                )
                
                audio_array = wav.squeeze(0).numpy()
                audio_segments.append(audio_array)
                
                # Add minimal natural pause between chunks
                if natural_pauses and i < len(chunks):
                    pause_samples = int(pause_duration * self.model.sr)
                    pause = np.zeros(pause_samples, dtype=np.float32)
                    audio_segments.append(pause)
                
            except Exception as e:
                print(f"❌ Error in chunk {i}: {e}")
                # Continue with next chunk instead of failing
                continue
        
        if not audio_segments:
            raise RuntimeError("Failed to generate any audio chunks")
        
        # Combine all segments
        combined_audio = np.concatenate(audio_segments)
        total_duration = len(combined_audio) / self.model.sr
        
        print(f"✅ Generated {total_duration:.1f} seconds of audio from {len(chunks)} chunks")
        
        return (self.model.sr, combined_audio)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the input text"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        # Remove multiple newlines but keep paragraph breaks
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        return cleaned
    
    def _set_seed(self, seed: int):
        """Set random seed"""
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _generate_chunk(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        temperature: float,
        cfg_weight: float
    ) -> torch.Tensor:
        """Generate audio for a single chunk"""
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path
        
        return self.model.generate(
            text,
            language_id=language_id,
            **generate_kwargs
        )
    
    def save_audio(self, audio_data: Tuple[int, np.ndarray], output_path: str) -> str:
        """Save generated audio to file"""
        sample_rate, audio_array = audio_data
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio_array, sample_rate)
        
        # Return stats
        duration = len(audio_array) / sample_rate
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        return f"💾 Saved: {output_path}\n⏱️ Duration: {duration:.1f}s\n📁 Size: {file_size:.1f} MB"


def test_very_long_text():
    """Test with a very long Italian text"""
    
    generator = OptimizedLongTextGenerator()
    
    # Long Italian text with accents
    very_long_text = """
    La storia dell'Italia è lunga e complessa, caratterizzata da una ricchezza culturale straordinaria 
    che si è sviluppata nel corso di millenni. Dall'antica Roma all'Impero Romano, dalle città-stato 
    medievali al Rinascimento, ogni epoca ha lasciato tracce indelebili nel patrimonio artistico e culturale del paese.
    
    Durante il periodo romano, l'Italia era il centro di un impero che si estendeva dall'Atlantico 
    all'Oceano Indiano. Le città come Roma, Milano, Napoli e Firenze erano centri di commercio, 
    arte e cultura. Gli antichi romani costruirono strade, acquedotti, teatri e anfiteatri che 
    ancora oggi testimoniano la loro grandezza ingegneristica.
    
    Il Rinascimento italiano, tra il XIV e il XVI secolo, fu un periodo di straordinaria creatività 
    artistica e intellettuale. Artisti come Leonardo da Vinci, Michelangelo, Raffaello e Botticelli 
    crearono opere d'arte che sono considerate tra le più belle e importanti della storia dell'umanità. 
    Città come Firenze, Roma e Venezia divennero centri di innovazione artistica e scientifica.
    
    L'Italia moderna è nata nel 1861 con l'unificazione, un processo lungo e complesso che ha portato 
    alla creazione dello stato italiano contemporaneo. Oggi l'Italia è famosa in tutto il mondo per 
    la sua cucina, la sua moda, i suoi paesaggi mozzafiato e il suo patrimonio artistico e culturale.
    
    La lingua italiana, con le sue caratteristiche lettere accentate come è, à, ì, ò, ù, riflette 
    la melodiosità e l'espressività del popolo italiano. Parole come città, università, così, più, 
    perché, caffè sono esempi della bellezza sonora di questa lingua romanza.
    
    La cucina italiana è probabilmente una delle più amate al mondo. Pasta, pizza, risotto, gelato 
    sono solo alcuni dei piatti che hanno conquistato i palati di tutto il mondo. Ogni regione ha 
    le sue specialità: dalla pasta alla carbonara del Lazio, al pesto ligure, dalla pizza napoletana 
    ai tortellini emiliani.
    """
    
    try:
        print("🎬 Testing very long text generation...")
        
        audio_data = generator.generate_long_text_audio(
            text=very_long_text,
            language_id="it",
            exaggeration=0.4,  # Slightly less exaggerated for natural speech
            temperature=0.7,   # Slightly lower for consistency
            cfg_weight=0.4,    # Lower for more natural flow
            max_chunk_chars=1000,  # Optimal chunk size
            natural_pauses=True,
            pause_duration=0.05  # Very short pauses
        )
        
        output_file = "test_very_long_italian.wav"
        result = generator.save_audio(audio_data, output_file)
        print(result)
        
        print("🎉 Very long text test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_very_long_text()
