from dataclasses import dataclass
from pathlib import Path
import os
import re

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=device, weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def _clean_markdown(self, text: str) -> str:
        """
        Remove Markdown formatting from text before TTS generation.
        Converts Markdown to plain text while preserving content.
        
        Args:
            text: Input text with potential Markdown formatting
            
        Returns:
            Clean text without Markdown syntax
        """
        # Remove Markdown headers (# ## ###)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic markers (**text** or *text*)
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        
        # Remove inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Replace multiple newlines with double newline (preserve paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using common punctuation marks."""
        # Split by sentence-ending punctuation and newlines
        sentences = re.split(r'[\.\!\?\n]', text)
        # Filter out empty strings and strip whitespace
        filtered_sentences = [s.strip() for s in sentences if s.strip()]
        return filtered_sentences
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs (preserves more narrative context).
        Paragraphs are separated by double newlines or multiple newlines.
        """
        # Split by double newline or more
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty strings and strip whitespace
        filtered_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return filtered_paragraphs
    
    def _split_into_chunks(self, text: str, max_sentences_per_chunk: int = 3) -> list[str]:
        """
        Split text into chunks of N sentences each.
        This preserves context better than single sentences while keeping chunks manageable.
        
        Args:
            text: Input text to split
            max_sentences_per_chunk: Maximum number of sentences per chunk (default: 3)
            
        Returns:
            List of text chunks, each containing up to max_sentences_per_chunk sentences
        """
        # First split into sentences
        sentences = self._split_into_sentences(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            if len(current_chunk) >= max_sentences_per_chunk:
                # Join sentences with period and space
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
        
        # Add remaining sentences if any
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _split_into_adaptive_chunks(self, text: str, target_chars: int = 800) -> list[str]:
        """
        Split text into chunks dynamically based on character count.
        Groups sentences together until reaching the target character count WITHOUT EXCEEDING IT.
        This maintains narrative context while optimizing generation efficiency.
        
        Args:
            text: Input text to split
            target_chars: Maximum character count per chunk (default: 800)
                         Each chunk will have UP TO target_chars, never exceeding it
            
        Returns:
            List of text chunks, each <= target_chars length
            
        Examples:
            >>> text = "First sentence. Second sentence. Third sentence. Fourth."
            >>> chunks = _split_into_adaptive_chunks(text, target_chars=50)
            >>> # Returns chunks where each chunk <= 50 chars
            >>> # Sentences are grouped until adding another would exceed limit
        """
        # First split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the target
            # We add +2 for ". " separator between sentences
            separator_length = 2 if current_chunk else 0
            new_total = current_length + separator_length + sentence_length
            
            if new_total > target_chars:
                # Would exceed target, save current chunk first (if not empty)
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = []
                    current_length = 0
                
                # Start new chunk with current sentence
                # If single sentence exceeds target, we still add it (can't split sentences)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Safe to add this sentence to current chunk
                current_chunk.append(sentence)
                current_length = new_total
        
        # Add remaining sentences if any
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _estimate_token_count(self, text: str, language_id: str = "en") -> int:
        """
        Estimate the number of tokens a text will generate.
        Rough approximation: ~50 tokens per sentence for speech generation.
        
        Args:
            text: Input text
            language_id: Language code
            
        Returns:
            Estimated token count
        """
        # Count sentences as a rough proxy
        num_sentences = len(self._split_into_sentences(text))
        
        # Average tokens per sentence varies by language and text complexity
        # Italian/English narrative: ~40-60 tokens per sentence
        # Short sentences: ~20-30 tokens
        # Long complex sentences: ~80-100 tokens
        avg_tokens_per_sentence = 50
        
        # Add some buffer for safety (20%)
        estimated_tokens = int(num_sentences * avg_tokens_per_sentence * 1.2)
        
        return estimated_tokens
    
    def _get_sentence_pause_duration(self, sentence: str, default_ms: int = 400) -> int:
        """
        Calculate natural pause duration after a sentence based on punctuation.
        
        Research-based pause durations:
        - Period (.): 400-500ms (declarative sentence)
        - Exclamation (!): 450-550ms (emphasis requires longer pause)
        - Question (?): 450-550ms (rising intonation needs recovery time)
        - Ellipsis (...): 600-800ms (indicates trailing thought)
        
        Args:
            sentence: The sentence text (original, before split)
            default_ms: Default pause if punctuation not detected
            
        Returns:
            Pause duration in milliseconds
        """
        # Look at the last character to determine punctuation type
        # Note: sentence is already split, so we need to check the original text
        # For now, use a natural default for period-ended sentences
        
        # Natural speech pause after period: 400-500ms
        # We use 400ms as a conservative, natural-sounding baseline
        # This is based on phonetic research showing inter-sentence pauses
        # average 300-600ms in natural speech, with 400ms being most common
        return default_ms

    def _generate_single(
        self,
        text,
        language_id,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
    ):
        """Generate audio for a single sentence/text chunk."""
        # Norm and tokenize text
        text = punc_norm(text)
        
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        auto_split=True,
        split_mode="adaptive",  # "sentences", "paragraphs", "chunks", "adaptive", or None
        chunk_size=3,  # sentences per chunk when split_mode="chunks"
        target_chars=800,  # NEW: target characters per chunk when split_mode="adaptive"
        sentence_pause_ms=400,
        max_new_tokens=2000,
    ):
        """
        Generate speech from text.
        
        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., 'en', 'it', 'fr')
            audio_prompt_path: Path to reference audio file
            exaggeration: Speech expressiveness (0.25-2.0, neutral=0.5)
            cfg_weight: CFG/Pace weight (0.2-1.0)
            temperature: Generation randomness (0.05-5.0)
            repetition_penalty: Penalty for repeated tokens
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            auto_split: If True, automatically split long text (uses split_mode)
            split_mode: How to split text:
                        - "adaptive" (default): Groups sentences to reach ~target_chars (best for books)
                        - "sentences": Split by punctuation (good for short texts)
                        - "paragraphs": Split by double newlines (preserves more context)
                        - "chunks": Group N sentences together (fixed-size chunks)
                        - None: Generate entire text at once (use with caution)
            chunk_size: Number of sentences per chunk when split_mode="chunks" (default: 3)
            target_chars: Target characters per chunk when split_mode="adaptive" (default: 800)
                         Recommended range: 600-1000 chars for optimal context/efficiency balance
            sentence_pause_ms: Duration of pause between segments in milliseconds (default: 400ms)
            max_new_tokens: Maximum tokens to generate per segment (default: 2000)
        
        Returns:
            torch.Tensor: Generated audio waveform
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        # Clean Markdown formatting before processing
        original_length = len(text)
        text = self._clean_markdown(text)
        cleaned_length = len(text)
        if cleaned_length < original_length:
            print(f"🧹 Cleaned Markdown formatting: {original_length} → {cleaned_length} characters")
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Auto-split text into segments if enabled
        if auto_split and split_mode:
            print(f"\n{'='*60}")
            print(f"🔤 Text Splitting Configuration")
            print(f"{'='*60}")
            print(f"Split mode: {split_mode}")
            print(f"Text length: {len(text)} characters")
            
            # Choose splitting method based on split_mode
            if split_mode == "sentences":
                print(f"📍 Splitting by individual sentences...")
                segments = self._split_into_sentences(text)
                segment_type = "sentences"
            elif split_mode == "paragraphs":
                print(f"📄 Splitting by paragraphs (double newlines)...")
                segments = self._split_into_paragraphs(text)
                segment_type = "paragraphs"
            elif split_mode == "chunks":
                print(f"📦 Splitting into fixed chunks of {chunk_size} sentences each...")
                segments = self._split_into_chunks(text, max_sentences_per_chunk=chunk_size)
                segment_type = f"{chunk_size}-sentence chunks"
            elif split_mode == "adaptive":
                print(f"🎯 Adaptive splitting with target of {target_chars} chars per chunk...")
                print(f"   (Will never exceed {target_chars} chars, concatenating sentences optimally)")
                segments = self._split_into_adaptive_chunks(text, target_chars=target_chars)
                segment_type = f"adaptive chunks (max {target_chars} chars)"
                # Log chunk sizes for verification
                chunk_sizes = [len(seg) for seg in segments]
                avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
                min_size = min(chunk_sizes) if chunk_sizes else 0
                max_size = max(chunk_sizes) if chunk_sizes else 0
                print(f"\n📊 Chunk Statistics:")
                print(f"   Total chunks: {len(segments)}")
                print(f"   Average size: {avg_size:.0f} chars")
                print(f"   Size range: {min_size}-{max_size} chars")
                print(f"   Target was: {target_chars} chars (max limit)")
                if max_size > target_chars:
                    print(f"   ⚠️  Note: {max_size} > {target_chars} because single sentence exceeds limit")
            else:
                # Invalid split_mode, treat as no split
                print(f"⚠️  Unknown split_mode '{split_mode}', processing as full text...")
                segments = [text]
                segment_type = "full text"
            
            print(f"{'='*60}\n")
            
            # If only one segment or text is short, generate directly
            if len(segments) <= 1:
                print(f"ℹ️  Only 1 segment detected, generating directly without concatenation...\n")
                return self._generate_single(
                    text=text,
                    language_id=language_id,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            
            # Estimate tokens and warn if necessary
            estimated_tokens = self._estimate_token_count(text, language_id)
            print(f"Estimated total tokens: ~{estimated_tokens}")
            if estimated_tokens > max_new_tokens and split_mode == "chunks":
                print(f"⚠️  Warning: Text may exceed max_new_tokens ({max_new_tokens}). Consider increasing it or using smaller chunks.")
            
            # Calculate pause duration BEFORE using it
            pause_duration_ms = self._get_sentence_pause_duration(text, default_ms=sentence_pause_ms)
            pause_samples = int(self.sr * pause_duration_ms / 1000)
            silence_pause = torch.zeros(1, pause_samples)
            
            # Generate audio for each segment and concatenate with pauses
            print(f"🎙️  Starting generation of {len(segments)} {segment_type}...")
            print(f"Pause between segments: {pause_duration_ms}ms\n")
            combined_audio = None
            
            for i, segment in enumerate(segments):
                # Show preview of segment (truncate if too long)
                preview = segment[:80] + "..." if len(segment) > 80 else segment
                print(f"Generating segment {i+1}/{len(segments)}: {preview}")
                
                wav = self._generate_single(
                    text=segment,
                    language_id=language_id,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                
                if combined_audio is None:
                    combined_audio = wav
                else:
                    # Add a natural pause between segments
                    combined_audio = torch.cat((combined_audio, silence_pause, wav), dim=1)
            
            print("\n✅ Audio generation complete.\n")
            return combined_audio
        else:
            # Generate without splitting
            print(f"\n{'='*60}")
            print(f"🔤 Text Splitting: DISABLED")
            print(f"{'='*60}")
            print(f"Text length: {len(text)} characters")
            print(f"Mode: Single-pass generation")
            print(f"Max tokens: {max_new_tokens}")
            if len(text) > 1000:
                print(f"⚠️  Warning: Long text ({len(text)} chars) without splitting may:")
                print(f"   - Take longer to generate")
                print(f"   - Risk exceeding max_new_tokens")
                print(f"   - Lose narrative context consistency")
                print(f"   Consider using auto_split=True with split_mode='adaptive'")
            print(f"{'='*60}\n")
            
            return self._generate_single(
                text=text,
                language_id=language_id,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
