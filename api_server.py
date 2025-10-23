"""
REST API server for Chatterbox Multilingual TTS
Provides endpoints to generate speech from text and return WAV audio files.
"""
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import torch
import torchaudio as ta
import io
import logging
from pathlib import Path

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

# Global model instance (loaded once at startup)
model = None
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_AUDIO_PROMPT = "pannofino.mp3"  # Change this to your default reference audio


class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=10000)
    language_id: str = Field(default="it", description=f"Language code. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}")
    audio_prompt_path: Optional[str] = Field(default=None, description="Path to reference audio file for voice cloning")
    cfg_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Classifier-free guidance weight (0.0-1.0)")
    exaggeration: float = Field(default=1.5, ge=0.5, le=3.0, description="Exaggeration factor for expressiveness (0.5-3.0)")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature (0.1-2.0)")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0, description="Repetition penalty (1.0-2.0)")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum probability threshold (0.0-1.0)")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling (0.0-1.0)")
    auto_split: bool = Field(default=True, description="Automatically split text into sentences")
    sentence_pause_ms: int = Field(default=400, ge=0, le=2000, description="Pause duration between sentences in milliseconds")


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
            "/health": "GET - Health check",
            "/languages": "GET - List supported languages"
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


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text and return as WAV file
    
    Returns a binary WAV audio file that can be directly played or saved.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate language
    if request.language_id not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language_id}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # Determine audio prompt path
    audio_prompt_path = request.audio_prompt_path or DEFAULT_AUDIO_PROMPT
    
    # Check if audio prompt exists
    if not Path(audio_prompt_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio prompt file not found: {audio_prompt_path}"
        )
    
    try:
        logger.info(f"Generating speech for text: '{request.text[:50]}...' in language: {request.language_id}")
        
        # Generate speech
        wav = model.generate(
            request.text,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration,
            language_id=request.language_id,
            audio_prompt_path=audio_prompt_path,
            auto_split=request.auto_split,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
            sentence_pause_ms=request.sentence_pause_ms
        )
        
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
                "Content-Disposition": "attachment; filename=generated_speech.wav"
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
        port=8000,
        reload=False,  # Set to True during development
        log_level="info"
    )
