"""Model management and TTS operations for the audiobook generation system."""

import torch
import random
import numpy as np
from chatterbox.tts import ChatterboxTTS
from typing import Any, Tuple, Optional


# Global device setting - will be imported from main file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTI_VOICE_DEVICE = "cpu"  # Force CPU for multi-voice processing


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible generation.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model() -> ChatterboxTTS:
    """Load TTS model for the default device.
    
    Returns:
        ChatterboxTTS: Loaded TTS model
    """
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def load_model_cpu() -> ChatterboxTTS:
    """Load model specifically for CPU processing.
    
    Returns:
        ChatterboxTTS: CPU-loaded TTS model
    """
    model = ChatterboxTTS.from_pretrained("cpu")
    return model


def clear_gpu_memory() -> None:
    """Clear GPU memory cache to prevent CUDA errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_memory() -> str:
    """Check current GPU memory usage.
    
    Returns:
        str: GPU memory status information
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return f"GPU Memory - Allocated: {allocated//1024//1024}MB, Cached: {cached//1024//1024}MB"
    return "CUDA not available"


def generate(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    seed_num: int, 
    cfgw: float
) -> Tuple[int, np.ndarray]:
    """Generate audio from text using the TTS model.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter for generation
        temperature: Temperature for generation randomness
        seed_num: Random seed (0 for random)
        cfgw: CFG weight parameter
        
    Returns:
        tuple: (sample_rate, audio_array)
    """
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())


def generate_with_cpu_fallback(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    cfg_weight: float
) -> Tuple[Any, str]:
    """Generate audio with automatic CPU fallback for problematic CUDA errors.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter
        temperature: Temperature parameter
        cfg_weight: CFG weight parameter
        
    Returns:
        tuple: (audio_wav, device_used)
    """
    # First try GPU if available
    if DEVICE == "cuda":
        try:
            clear_gpu_memory()
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            return wav, "GPU"
        except RuntimeError as e:
            if ("srcIndex < srcSelectDimSize" in str(e) or 
                "CUDA" in str(e) or 
                "out of memory" in str(e).lower()):
                
                print(f"⚠️ CUDA error detected, falling back to CPU: {str(e)[:100]}...")
                # Fall through to CPU mode
            else:
                raise e
    
    # CPU fallback or primary CPU mode
    try:
        # Load CPU model if needed
        cpu_model = ChatterboxTTS.from_pretrained("cpu")
        wav = cpu_model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        return wav, "CPU"
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU generation failed: {str(e)}")


def generate_with_retry(
    model: ChatterboxTTS, 
    text: str, 
    audio_prompt_path: str, 
    exaggeration: float, 
    temperature: float, 
    cfg_weight: float, 
    max_retries: int = 3
) -> Tuple[Any, str]:
    """Generate audio with retry mechanism for robustness.
    
    Args:
        model: TTS model instance
        text: Text to convert to speech
        audio_prompt_path: Path to audio prompt file
        exaggeration: Exaggeration parameter
        temperature: Temperature parameter
        cfg_weight: CFG weight parameter
        max_retries: Maximum number of retry attempts
        
    Returns:
        tuple: (audio_wav, device_used)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return generate_with_cpu_fallback(
                model, text, audio_prompt_path, exaggeration, temperature, cfg_weight
            )
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                clear_gpu_memory()
    
    raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")


def force_cpu_processing() -> bool:
    """Check if we should force CPU processing for stability.
    
    Returns:
        bool: True if CPU processing should be forced
    """
    # For multi-voice, always use CPU to avoid CUDA indexing issues
    return True


def get_model_device_str(model_obj: Optional[ChatterboxTTS]) -> str:
    """Get the device string for a model object.
    
    Args:
        model_obj: TTS model instance
        
    Returns:
        str: Device information string
    """
    if model_obj is None:
        return "No model loaded"
    
    try:
        # Try to access model device info
        if hasattr(model_obj, 'device'):
            return f"Model device: {model_obj.device}"
        elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'device'):
            return f"Model device: {model_obj.model.device}"
        else:
            return "Device info unavailable"
    except Exception as e:
        return f"Error getting device info: {str(e)}" 