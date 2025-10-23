"""
REST API server for Chatterbox Multilingual TTS
Provides endpoints to generate speech from text and return WAV audio files.
"""
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
import torchaudio as ta
import io
import logging
import urllib.request
from pathlib import Path
import hashlib
import json

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-Speech API using Chatterbox Multilingual TTS",
    version="1.0.0"
)


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Provide detailed validation error messages"""
    errors = exc.errors()
    logger.error(f"Validation error: {errors}")
    
    # Format errors in a readable way
    error_details = []
    for error in errors:
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_type = error["type"]
        error_details.append(f"{field}: {message} (type: {error_type})")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": error_details,
            "received_body": await request.body() if await request.body() else None
        }
    )

# Global model instance (loaded once at startup)
model = None
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache directory for generated audio
CACHE_DIR = Path("audio_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Language-specific default audio prompts (from multilingual_app.py)
LANGUAGE_AUDIO_PROMPTS = {
    "it": "pannofino.mp3",  # Local file
    "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_m1.flac",
    "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_f1.flac",
}


def download_audio_if_needed(audio_path: str) -> str:
    """
    Download audio file from URL if needed, otherwise return the path as-is.
    Cached downloads are stored in temp_audio/ directory.
    """
    if audio_path.startswith(('http://', 'https://')):
        # Create a temp directory if it doesn't exist
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract filename from URL
        filename = audio_path.split('/')[-1]
        local_path = temp_dir / filename
        
        # Download if not already cached
        if not local_path.exists():
            logger.info(f"Downloading audio from {audio_path}...")
            try:
                urllib.request.urlretrieve(audio_path, local_path)
                logger.info(f"Downloaded to {local_path}")
            except Exception as e:
                logger.error(f"Failed to download audio: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download audio prompt: {str(e)}")
        else:
            logger.info(f"Using cached audio: {local_path}")
        
        return str(local_path)
    return audio_path


def get_default_audio_prompt(language_id: str) -> str:
    """
    Get the default audio prompt for a given language.
    Downloads remote files if needed and caches them locally.
    """
    if language_id not in LANGUAGE_AUDIO_PROMPTS:
        logger.warning(f"No default audio prompt for language '{language_id}', using Italian")
        language_id = "it"
    
    audio_path = LANGUAGE_AUDIO_PROMPTS[language_id]
    return download_audio_if_needed(audio_path)


class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=1000000)  # Increased to 1M chars (~400 pages)
    language_id: str = Field(default="it", description=f"Language code. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}")
    audio_prompt_path: Optional[str] = Field(default=None, description="Path to reference audio file for voice cloning")
    cfg_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Classifier-free guidance weight (0.0-1.0)")
    exaggeration: float = Field(default=1.5, ge=0.5, le=3.0, description="Exaggeration factor for expressiveness (0.5-3.0)")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature (0.1-2.0)")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0, description="Repetition penalty (1.0-2.0)")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum probability threshold (0.0-1.0)")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling (0.0-1.0)")
    auto_split: bool = Field(default=True, description="Automatically split text into sentences")
    split_mode: str = Field(default="adaptive", description="Split mode: 'adaptive' (dynamic by chars), 'sentences', 'paragraphs', 'chunks', or 'none'")
    chunk_size: int = Field(default=3, ge=1, le=10, description="Sentences per chunk when split_mode='chunks'")
    target_chars: int = Field(default=800, ge=200, le=2000, description="Target characters per chunk when split_mode='adaptive'")
    sentence_pause_ms: int = Field(default=400, ge=0, le=2000, description="Pause duration between sentences in milliseconds")
    max_new_tokens: int = Field(default=2000, ge=100, le=5000, description="Maximum tokens to generate per segment")


def generate_cache_key(request: TTSRequest, audio_prompt_path: str) -> str:
    """
    Generate a unique hash key for caching based on all generation parameters.
    Returns the hash string that can be used as a filename.
    """
    # Create a dictionary with all parameters that affect the output
    cache_params = {
        "text": request.text,
        "language_id": request.language_id,
        "audio_prompt_path": audio_prompt_path,
        "cfg_weight": request.cfg_weight,
        "exaggeration": request.exaggeration,
        "temperature": request.temperature,
        "repetition_penalty": request.repetition_penalty,
        "min_p": request.min_p,
        "top_p": request.top_p,
        "auto_split": request.auto_split,
        "split_mode": request.split_mode,
        "chunk_size": request.chunk_size,
        "target_chars": request.target_chars,
        "sentence_pause_ms": request.sentence_pause_ms,
        "max_new_tokens": request.max_new_tokens
    }
    
    # Convert to JSON string (sorted keys for consistency)
    params_str = json.dumps(cache_params, sort_keys=True)
    
    # Generate SHA256 hash
    hash_obj = hashlib.sha256(params_str.encode('utf-8'))
    return hash_obj.hexdigest()


def get_cached_audio(cache_key: str) -> Optional[Path]:
    """
    Check if cached audio exists for the given cache key.
    Returns the Path to the cached file if it exists, None otherwise.
    """
    cache_file = CACHE_DIR / f"{cache_key}.wav"
    if cache_file.exists():
        logger.info(f"✅ Cache HIT: {cache_key[:16]}... (file exists)")
        return cache_file
    logger.info(f"❌ Cache MISS: {cache_key[:16]}... (generating new audio)")
    return None


def save_to_cache(cache_key: str, wav_tensor: torch.Tensor, sample_rate: int) -> Path:
    """
    Save generated audio to cache.
    Returns the Path to the saved file.
    """
    cache_file = CACHE_DIR / f"{cache_key}.wav"
    ta.save(str(cache_file), wav_tensor, sample_rate, format="wav")
    logger.info(f"💾 Saved to cache: {cache_key[:16]}... ({cache_file})")
    return cache_file


@app.on_event("startup")
async def startup_event():
    """Load the TTS model on startup"""
    global model
    logger.info(f"Loading Chatterbox Multilingual TTS model on {MODEL_DEVICE}...")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=MODEL_DEVICE)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Chatterbox TTS API",
        "version": "1.0.0",
        "status": "running",
        "device": MODEL_DEVICE,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "endpoints": {
            "/generate": "POST - Generate speech from text",
            "/generate-simple": "POST - Simple generation with query params",
            "/validate": "POST - Validate request without generating",
            "/health": "GET - Health check",
            "/languages": "GET - List supported languages",
            "/cache/stats": "GET - Cache statistics",
            "/cache/clear": "DELETE - Clear cache"
        }
    }


@app.post("/validate")
async def validate_request(request: TTSRequest):
    """
    Validate a TTS request without actually generating audio.
    Useful for debugging 422 errors.
    """
    return {
        "status": "valid",
        "message": "Request is valid and ready for generation",
        "received_parameters": {
            "text_length": len(request.text),
            "language_id": request.language_id,
            "audio_prompt_path": request.audio_prompt_path,
            "cfg_weight": request.cfg_weight,
            "exaggeration": request.exaggeration,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "min_p": request.min_p,
            "top_p": request.top_p,
            "auto_split": request.auto_split,
            "sentence_pause_ms": request.sentence_pause_ms
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": MODEL_DEVICE
    }


@app.get("/languages")
async def list_languages():
    """List all supported languages"""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "total": len(SUPPORTED_LANGUAGES)
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    cache_files = list(CACHE_DIR.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "cache_dir": str(CACHE_DIR),
        "total_files": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_files": [
            {
                "hash": f.stem[:16] + "...",
                "size_kb": round(f.stat().st_size / 1024, 2),
                "created": f.stat().st_ctime
            }
            for f in sorted(cache_files, key=lambda x: x.stat().st_ctime, reverse=True)[:10]
        ]
    }


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached audio files"""
    cache_files = list(CACHE_DIR.glob("*.wav"))
    deleted_count = 0
    deleted_size = 0
    
    for f in cache_files:
        deleted_size += f.stat().st_size
        f.unlink()
        deleted_count += 1
    
    return {
        "status": "success",
        "deleted_files": deleted_count,
        "freed_space_mb": round(deleted_size / (1024 * 1024), 2)
    }


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text and return as WAV file
    
    Returns a binary WAV audio file that can be directly played or saved.
    """
    logger.info(f"Received request: text_length={len(request.text)}, language={request.language_id}, cfg={request.cfg_weight}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate language
    if request.language_id not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language_id}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # Determine audio prompt path
    # If user provides a custom path, use it; otherwise get the default for the language
    if request.audio_prompt_path:
        audio_prompt_path = download_audio_if_needed(request.audio_prompt_path)
    else:
        audio_prompt_path = get_default_audio_prompt(request.language_id)
    
    # Check if audio prompt exists
    if not Path(audio_prompt_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio prompt file not found: {audio_prompt_path}"
        )
    
    # Generate cache key based on all parameters
    cache_key = generate_cache_key(request, audio_prompt_path)
    logger.info(f"🔑 Cache key: {cache_key[:16]}...")
    
    # Check if cached audio exists
    cached_file = get_cached_audio(cache_key)
    
    if cached_file:
        # Return cached file directly
        logger.info(f"📤 Returning cached audio from {cached_file}")
        
        # Read the cached WAV file
        with open(cached_file, 'rb') as f:
            audio_data = f.read()
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav",
                "X-Cache-Status": "HIT"
            }
        )
    
    try:
        logger.info(f"Generating speech for text: '{request.text[:50]}...' in language: {request.language_id}")
        logger.info(f"Split mode: {request.split_mode}, target_chars: {request.target_chars if request.split_mode == 'adaptive' else 'N/A'}")
        
        # Generate speech
        wav = model.generate(
            request.text,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration,
            language_id=request.language_id,
            audio_prompt_path=audio_prompt_path,
            auto_split=request.auto_split,
            split_mode=request.split_mode if request.split_mode != "none" else None,
            chunk_size=request.chunk_size,
            target_chars=request.target_chars,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
            sentence_pause_ms=request.sentence_pause_ms,
            max_new_tokens=request.max_new_tokens
        )
        
        # Save to cache
        save_to_cache(cache_key, wav, model.sr)
        
        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()
        
        # Save the audio to the buffer as WAV format
        ta.save(buffer, wav, model.sr, format="wav")
        
        # Seek to the beginning of the buffer
        buffer.seek(0)
        
        logger.info(f"Speech generated successfully. Audio duration: {wav.shape[1] / model.sr:.2f}s")
        
        # Return the WAV file as a streaming response
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_speech.wav",
                "X-Cache-Status": "MISS"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.post("/generate-simple")
async def generate_speech_simple(text: str, language_id: str = "it"):
    """
    Simplified endpoint that accepts just text and language as query parameters.
    Uses default settings for all other parameters.
    
    Example: POST /generate-simple?text=Ciao mondo&language_id=it
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create request with defaults
    request = TTSRequest(text=text, language_id=language_id)
    
    # Call the main generate endpoint
    return await generate_speech(request)


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,  # Set to True during development
        log_level="info"
    )
