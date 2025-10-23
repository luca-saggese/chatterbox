# Investigation: Echo/Reverb Artifact at End of Generation

## Problem Description

User reports hearing an echo-like artifact at the end of generated audio segments. This is NOT:
- ❌ Concatenation noise between sentences (different issue, already addressed)
- ❌ Clicks/pops (also different)

This IS:
- ✅ Echo/reverb sound within the generated audio itself
- ✅ Appears at the end of each sentence generation
- ✅ Sounds like audio is "bleeding through" or continuing after speech ends

## Hypothesis

The echo is likely caused by one of these issues:

### 1. **Forced EOS Tokens Creating Decoder Artifacts** (Most Likely)

The `AlignmentStreamAnalyzer` can force EOS tokens when it detects:
- Long tail (audio continuing after text is complete)
- Alignment repetition (attention stuck on previous tokens)
- Token repetition (same token repeated multiple times)

**Problem**: When EOS is forced, the model's internal state may be inconsistent:
```python
# From alignment_stream_analyzer.py line 181-185
if long_tail or alignment_repetition or token_repetition:
    logger.warning(f"forcing EOS token...")
    logits = -(2**15) * torch.ones_like(logits)  # Force all tokens to -inf
    logits[..., self.eos_idx] = 2**15              # Except EOS to +inf
```

This hard override might cause:
- The decoder (S3Gen) to receive tokens that don't match the actual speech content
- The HiFiGAN vocoder to produce artifacts from inconsistent mel-spectrograms
- Reference audio "spillover" extending beyond intended speech

### 2. **Reference Audio Spillover**

The S3Gen model uses `trim_fade` (40ms) to fade in and reduce reference audio spillover:
```python
# From s3gen.py line 257
output_wavs[:, :len(self.trim_fade)] *= self.trim_fade
```

**Problem**: This only handles the START, not the END. If reference audio is bleeding through at the end, it would sound like echo/reverb.

### 3. **Missing EOS Token in Generation**

If the model generates tokens beyond the actual speech content (because EOS wasn't properly detected or generated), the decoder might:
- Continue producing audio based on "empty" or invalid tokens
- Create reverb-like artifacts from the decoder's extrapolation

### 4. **HiFiGAN Vocoder Artifacts**

The HiFiGAN vocoder might be producing artifacts when:
- Input mel-spectrograms have discontinuities
- The model receives "incomplete" or "cut-off" sequences
- Cache state is inconsistent (though currently cache is reset each time)

## Diagnostic Logging Added

To identify the root cause, I've added detailed logging:

### In `mtl_tts.py` (_generate_single):
```python
print(f"🔍 Raw tokens generated: {speech_tokens.shape[0]} tokens")
print(f"🔍 First 10 tokens: {speech_tokens[:10].tolist()}")
print(f"🔍 Last 10 tokens: {speech_tokens[-10:].tolist()}")
print(f"🔍 After drop_invalid_tokens: {speech_tokens.shape[0]} tokens (removed X)")
```

### In `t3.py` (inference):
```python
logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
logger.info(f"   Total tokens generated: {len(predicted)}")
logger.info(f"   Last 5 tokens: {...}")
```

### In `alignment_stream_analyzer.py`:
```python
logger.warning(f"🚨 forcing EOS token at frame {self.curr_frame_pos}...")
logger.warning(f"   Text position: {self.text_position}/{S}, Complete: {self.complete}...")
```

## Testing Instructions

1. **Generate a sentence and check logs**:
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

tts = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
tts.prepare_conditionals("reference.wav")

# Generate with logging
audio = tts.generate(
    text="Test sentence to check for echo at the end.",
    language_id="en",
    auto_split=False  # Test single sentence first
)
```

2. **Analyze the output**:
   - Check console for token information
   - Listen for echo at the end
   - Note when "forcing EOS" messages appear

3. **Key questions to answer**:
   - Is EOS being forced prematurely?
   - Are there many tokens after the last valid speech token?
   - Do the last tokens look suspicious (repetitive, out of vocab range)?
   - Does the echo duration correlate with number of tokens after natural speech ends?

## Expected Findings

### If EOS is forced too early:
```
🚨 forcing EOS token at frame 150...
✅ EOS token detected! Stopping generation at step 151
   Last 5 tokens: [1234, 1235, 1236, 6562, 6562]  <- EOS is 6562
```
→ The tokens before EOS might have created artifacts

### If tokens continue after speech ends:
```
🔍 Raw tokens generated: 200 tokens
🔍 Last 10 tokens: [1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 6562]
```
→ Too many tokens for the text length, creating extra audio

### If reference spillover:
```
🔍 First 10 tokens: [6561, 1234, 1235, ...]  <- SOS + immediate real tokens
```
→ Check if reference audio characteristics appear at the end

## Potential Fixes (After Diagnosis)

### Fix 1: If EOS forcing is the issue
- Make alignment analyzer less aggressive (already done partially)
- Add gradual fade instead of hard cutoff
- Better detection of "natural" end vs. artifacts

### Fix 2: If token overflow is the issue
- Limit max tokens based on text length
- Add better EOS detection
- Trim tokens after a "silence" pattern

### Fix 3: If reference spillover is the issue
- Apply trim_fade at the END as well as start
- Increase fade duration
- Better reference audio preprocessing

### Fix 4: If vocoder artifacts
- Add post-processing to decoder output
- Smooth mel-spectrogram transitions
- Check HiFiGAN cache state

## Next Steps

1. ✅ **Added diagnostic logging** (completed)
2. ⏳ **Run test generation and collect logs**
3. ⏳ **Analyze which scenario matches the observed behavior**
4. ⏳ **Implement targeted fix based on findings**
5. ⏳ **Verify fix resolves echo without side effects**

## Important Notes

- Do NOT add blanket fades/trims without understanding root cause
- The issue might be combination of multiple factors
- Fix should address the source, not mask symptoms
- Test with multiple languages and text lengths

---

**Status**: 🔍 Investigation phase - Diagnostic logging added  
**Date**: October 23, 2025  
**Next**: Run test generation and analyze logs
