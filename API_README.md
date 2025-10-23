# Chatterbox TTS REST API

A FastAPI-based REST API server for the Chatterbox Multilingual Text-to-Speech system.

## Features

- ✅ Generate speech from text via REST API
- ✅ Support for multiple languages (Italian, English, Spanish, French, German, Portuguese)
- ✅ Voice cloning with reference audio
- ✅ Automatic sentence splitting and natural pauses
- ✅ Configurable generation parameters (temperature, CFG weight, etc.)
- ✅ Returns audio as binary WAV file
- ✅ Health check and language listing endpoints

## Installation

1. Install API dependencies:
```bash
pip install -r requirements-api.txt
```

2. Make sure you have the Chatterbox model installed:
```bash
pip install -e .
```

## Quick Start

### 1. Start the Server

```bash
python api_server.py
```

The server will start on `http://localhost:8000`

### 2. Test the API

Open another terminal and run the test client:
```bash
python test_api_client.py
```

Or use curl:
```bash
# Simple request
curl -X POST "http://localhost:8000/generate-simple?text=Ciao%20mondo&language_id=it" \
  --output output.wav

# Full request with custom parameters
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ciao mondo! Come stai oggi?",
    "language_id": "it",
    "cfg_weight": 0.2,
    "exaggeration": 1.5,
    "auto_split": true
  }' \
  --output output.wav
```

### 3. View API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### GET `/`
Root endpoint with API information.

**Response:**
```json
{
  "name": "Chatterbox TTS API",
  "version": "1.0.0",
  "status": "running",
  "supported_languages": ["it", "en", "es", "fr", "de", "pt"]
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### GET `/languages`
List all supported languages.

**Response:**
```json
{
  "supported_languages": {
    "it": "Italian",
    "en": "English",
    ...
  },
  "total": 6
}
```

### POST `/generate`
Generate speech from text (full control over all parameters).

**Request Body:**
```json
{
  "text": "Your text here",
  "language_id": "it",
  "audio_prompt_path": "path/to/reference.mp3",
  "cfg_weight": 0.2,
  "exaggeration": 1.5,
  "temperature": 0.8,
  "repetition_penalty": 1.2,
  "min_p": 0.05,
  "top_p": 0.95,
  "auto_split": true,
  "sentence_pause_ms": 400
}
```

**Response:** Binary WAV audio file

**Parameters:**
- `text` (required): Text to convert to speech (1-10000 characters)
- `language_id`: Language code (default: "it")
- `audio_prompt_path`: Path to reference audio for voice cloning (optional)
- `cfg_weight`: Classifier-free guidance weight, 0.0-1.0 (default: 0.2)
- `exaggeration`: Expressiveness factor, 0.5-3.0 (default: 1.5)
- `temperature`: Sampling temperature, 0.1-2.0 (default: 0.8)
- `repetition_penalty`: Repetition penalty, 1.0-2.0 (default: 1.2)
- `min_p`: Minimum probability threshold, 0.0-1.0 (default: 0.05)
- `top_p`: Top-p sampling, 0.0-1.0 (default: 0.95)
- `auto_split`: Automatically split into sentences (default: true)
- `sentence_pause_ms`: Pause between sentences in ms, 0-2000 (default: 400)

### POST `/generate-simple`
Simplified endpoint with query parameters (uses defaults for other settings).

**Query Parameters:**
- `text` (required): Text to convert to speech
- `language_id`: Language code (default: "it")

**Example:**
```bash
curl -X POST "http://localhost:8000/generate-simple?text=Hello%20world&language_id=en" \
  --output output.wav
```

## Configuration

### Change Default Reference Audio

Edit `api_server.py` and change:
```python
DEFAULT_AUDIO_PROMPT = "pannofino.mp3"  # Your default reference audio
```

### Change Server Port

Edit `api_server.py` at the bottom:
```python
uvicorn.run(
    "api_server:app",
    host="0.0.0.0",
    port=8000,  # Change this
    ...
)
```

Or run with custom port:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 5000
```

### Enable GPU/CPU

The server automatically detects and uses CUDA if available. To force CPU:
```python
MODEL_DEVICE = "cpu"  # In api_server.py
```

## Python Client Example

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "text": "Ciao mondo! Questo è un test.",
        "language_id": "it",
        "cfg_weight": 0.2,
        "exaggeration": 1.5
    }
)

# Save the audio
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved!")
else:
    print(f"Error: {response.json()}")
```

## JavaScript/TypeScript Client Example

```javascript
// Using fetch API
async function generateSpeech(text, languageId = 'it') {
  const response = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      language_id: languageId,
      cfg_weight: 0.2,
      exaggeration: 1.5,
      auto_split: true
    })
  });
  
  if (response.ok) {
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    // Play the audio
    const audio = new Audio(url);
    audio.play();
    
    // Or download it
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated_speech.wav';
    a.click();
  } else {
    console.error('Error:', await response.json());
  }
}

// Usage
generateSpeech("Ciao mondo!", "it");
```

## Production Deployment

### Using Gunicorn (recommended for production)

```bash
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t chatterbox-api .
docker run -p 8000:8000 --gpus all chatterbox-api
```

### Environment Variables

You can configure the server using environment variables:

```bash
export CHATTERBOX_DEVICE=cuda  # or cpu
export CHATTERBOX_PORT=8000
export CHATTERBOX_WORKERS=4
export CHATTERBOX_DEFAULT_AUDIO=pannofino.mp3
```

## Monitoring

### Check Server Logs

The server logs all requests with timestamps and performance metrics:
```
INFO:     Generating speech for text: 'Ciao mondo...' in language: it
INFO:     Speech generated successfully. Audio duration: 2.34s
```

### Performance Metrics

Typical generation times:
- Short text (1 sentence): ~1-2 seconds on GPU
- Medium text (5 sentences): ~5-8 seconds on GPU
- Long text (20 sentences): ~20-30 seconds on GPU

CPU is significantly slower (3-5x).

## Troubleshooting

### Model not loading
**Error:** "Model not loaded"
**Solution:** Wait for the startup to complete. Check logs for loading errors.

### Out of memory
**Error:** CUDA out of memory
**Solution:** 
- Reduce batch processing
- Use CPU mode
- Reduce max_new_tokens parameter

### Audio prompt not found
**Error:** "Audio prompt file not found"
**Solution:** 
- Ensure the reference audio file exists
- Use absolute paths
- Or omit `audio_prompt_path` to use default

### Port already in use
**Error:** "Address already in use"
**Solution:**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
python api_server.py --port 8001
```

## License

Same license as the main Chatterbox project (MIT).

## Support

For issues and questions:
- Check the main Chatterbox README
- Review API documentation at `/docs`
- Enable debug logging: `uvicorn api_server:app --log-level debug`
