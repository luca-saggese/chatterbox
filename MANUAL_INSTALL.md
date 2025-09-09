# Manual Installation Guide

This guide is for users who prefer to install Chatterbox TTS manually instead of using the `install-audiobook.bat` file.

## Prerequisites

- **Python 3.10+** (Required)
- **NVIDIA GPU with CUDA support** (Recommended for best performance)
- **Git** (Optional, for updates)

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 4. Install Chatterbox TTS Package

```bash
# Install the Chatterbox TTS package in development mode
pip install -e .
```

## CPU-Only Installation

If you don't have an NVIDIA GPU or prefer CPU-only installation:

1. Edit `requirements.txt` and remove the `+cu121` suffixes from PyTorch packages
2. Remove or comment out the `--index-url` line
3. Install normally with `pip install -r requirements.txt`

**Modified PyTorch lines for CPU:**
```
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
```

## Troubleshooting

### Common Issues:

**PyTorch CUDA Issues:**
- Ensure NVIDIA drivers are up to date
- Verify CUDA compatibility with your GPU
- Try CPU installation if GPU issues persist

**Pydantic Compatibility:**
- The requirements.txt pins pydantic to version 2.10.6 for stability
- If you encounter issues, try: `pip install pydantic==2.10.6 --force-reinstall`

**Import Errors:**
- Make sure you're in the project root directory
- Verify the virtual environment is activated
- Run `pip install -e .` again if needed

### Verification

Test your installation:

```bash
# Test PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Test Chatterbox TTS
python -c "from chatterbox.tts import ChatterboxTTS; print('Chatterbox TTS imported successfully!')"
```

## Running the Application

After successful installation:

```bash
# Option 1: Use launcher scripts (recommended)
# For local-only access:
launch_local.bat

# For network access:
launch_network.bat

# For public sharing:
launch_huggingface.bat

# Option 2: Direct execution
python gradio_tts_app_audiobook.py
```

## Notes

- This manual installation provides the same functionality as `install-audiobook.bat`
- The batch file installer includes additional error checking and troubleshooting
- If you encounter issues, you can always fall back to using `install-audiobook.bat`
- Virtual environment is highly recommended to avoid dependency conflicts 