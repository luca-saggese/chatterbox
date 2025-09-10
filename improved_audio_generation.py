#!/usr/bin/env python3
"""
Improved audio generation functions to replace problematic chunking in audiobook app
"""

import torch
import numpy as np
import re
from typing import Optional, Tuple, List

def generate_long_audio_improved(
    model, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float = 0.5, 
    temperature: float = 0.8, 
    cfg_weight: float = 0.5, 
    min_p: float = 0.05, 
    top_p: float = 1.0, 
    repetition_penalty: float = 1.2, 
    language_id: str = "it",
    max_length: int = 1200,  # Increased max length per chunk
    add_smart_pauses: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Generate long audio using improved approach that minimizes chunking artifacts.
    
    This function uses the multilingual model's direct generation capability 
    instead of problematic line-by-line chunking that causes repetitions.
    
    Args:
        model: TTS model (ChatterboxMultilingualTTS or ChatterboxTTS)
        text: Input text to synthesize
        audio_prompt_path: Reference audio file path
        exaggeration: Speech expressiveness (0.25-2.0)
        temperature: Generation randomness (0.05-5.0)
        cfg_weight: CFG guidance weight (0.2-1.0)
        min_p: Minimum probability threshold
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        language_id: Language code (e.g., 'it', 'en')
        max_length: Maximum characters per chunk (only for very long texts)
        add_smart_pauses: Whether to add intelligent pauses
    
    Returns:
        Tuple of (sample_rate, audio_array)
    """
    
    if model is None:
        raise RuntimeError("Model is not loaded")
    
    sample_rate = getattr(model, "sr", 24000)
    
    # Clean up text
    text = text.strip()
    if not text:
        raise ValueError("Text is empty")
    
    print(f"🎤 Generating audio for {len(text)} characters using improved method")
    
    # For most texts, try single generation first (avoids chunking altogether)
    if len(text) <= max_length:
        print("📝 Using direct generation (no chunking)")
        try:
            return _generate_single_direct(
                model, text, audio_prompt_path, exaggeration, 
                temperature, cfg_weight, min_p, top_p, 
                repetition_penalty, language_id, sample_rate
            )
        except Exception as e:
            print(f"⚠️ Direct generation failed: {e}")
            print("🔄 Falling back to smart chunking...")
    
    # Smart chunking for very long texts
    print("📄 Using smart paragraph-based chunking")
    chunks = _smart_chunk_by_paragraphs(text, max_length)
    print(f"Split into {len(chunks)} chunks")
    
    audio_segments = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"🎵 Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)")
        
        try:
            _, chunk_audio = _generate_single_direct(
                model, chunk, audio_prompt_path, exaggeration,
                temperature, cfg_weight, min_p, top_p,
                repetition_penalty, language_id, sample_rate
            )
            audio_segments.append(chunk_audio)
            
            # Add smart pause between chunks
            if add_smart_pauses and i < len(chunks):
                pause_duration = _calculate_smart_pause(chunks[i-1], chunks[i] if i < len(chunks) else "")
                if pause_duration > 0:
                    pause_samples = int(pause_duration * sample_rate)
                    pause = np.zeros(pause_samples, dtype=np.float32)
                    audio_segments.append(pause)
                    print(f"🔇 Added {pause_duration:.2f}s pause")
            
        except Exception as e:
            print(f"❌ Error generating chunk {i}: {e}")
            # Add brief silence instead of failing
            fallback_samples = int(0.5 * sample_rate)
            fallback_audio = np.zeros(fallback_samples, dtype=np.float32)
            audio_segments.append(fallback_audio)
    
    # Combine all segments
    if audio_segments:
        combined_audio = np.concatenate(audio_segments)
        duration = len(combined_audio) / sample_rate
        print(f"✅ Generated {duration:.1f} seconds of audio")
        return (sample_rate, combined_audio)
    else:
        raise RuntimeError("Failed to generate any audio")


def _generate_single_direct(
    model, text: str, audio_prompt_path: str, exaggeration: float,
    temperature: float, cfg_weight: float, min_p: float, top_p: float,
    repetition_penalty: float, language_id: str, sample_rate: int
) -> Tuple[int, np.ndarray]:
    """Generate audio for a single text chunk using direct model call"""
    
    # Check if this is a multilingual model
    if hasattr(model, 'generate') and hasattr(model, 'get_supported_languages'):
        # Use multilingual model (like in multilingual_app.py)
        try:
            wav = model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                # Note: multilingual model may not support all parameters
            )
            return (sample_rate, wav.squeeze(0).numpy())
        except TypeError as e:
            # Some parameters might not be supported
            print(f"⚠️ Some parameters not supported by multilingual model: {e}")
            wav = model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return (sample_rate, wav.squeeze(0).numpy())
    
    else:
        # Use basic ChatterboxTTS
        conds = model.prepare_conditionals(audio_prompt_path, exaggeration)
        wav = model.generate(
            text,
            conds,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return (sample_rate, wav.squeeze(0).numpy())


def _smart_chunk_by_paragraphs(text: str, max_length: int) -> List[str]:
    """
    Smart chunking that respects paragraph and sentence boundaries
    Much better than line-by-line chunking that causes repetitions
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    
    # First, split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed max_length
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_length:
            # Save current chunk and start new one
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If any chunk is still too long, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            sentence_chunks = _split_by_sentences(chunk, max_length)
            final_chunks.extend(sentence_chunks)
    
    return final_chunks


def _split_by_sentences(text: str, max_length: int) -> List[str]:
    """Split text by sentences when paragraph chunking isn't enough"""
    sentences = re.split(r'([.!?]+\s*)', text)
    chunks = []
    current_chunk = ""
    
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
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _calculate_smart_pause(prev_chunk: str, next_chunk: str) -> float:
    """Calculate intelligent pause duration between chunks"""
    
    # Base pause
    base_pause = 0.3
    
    # Longer pause after paragraphs (indicated by double newlines in original)
    if prev_chunk.endswith('.') or prev_chunk.endswith('!') or prev_chunk.endswith('?'):
        # Check if this looks like end of paragraph/section
        if any(keyword in prev_chunk.lower().split()[-10:] for keyword in ['fine', 'conclusione', 'quindi', 'inoltre']):
            return base_pause + 0.4  # Longer pause
        return base_pause + 0.2  # Medium pause after sentences
    
    # Shorter pause for continuation
    return base_pause


# Replacement function for the existing generate() function
def generate_improved(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p=0.05, top_p=1.0, repetition_penalty=1.2, language_id="it"):
    """
    Improved replacement for the existing generate() function in audiobook app.
    This avoids the problematic line-by-line chunking that causes repetitions.
    """
    if model is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained("cuda" if torch.cuda.is_available() else "cpu")

    if seed_num != 0:
        torch.manual_seed(int(seed_num))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed_num))

    # Use the improved long audio generation
    sample_rate, audio_array = generate_long_audio_improved(
        model=model,
        text=text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        language_id=language_id,
        max_length=1200,  # Larger chunks for better continuity
        add_smart_pauses=True
    )
    
    return (sample_rate, audio_array)
