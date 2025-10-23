# Fix for Progressive Speech Acceleration Issue (GitHub #327)

## Problem Summary

During long speech generation (approaching 1000 tokens), the generated speech progressively speeds up, becoming unnaturally fast and sometimes skipping words or phonemes. The issue manifests as:

- Normal pace at the beginning of generation
- Gradual acceleration proportional to token count
- Significant speedup near max token limit (~1000 tokens)
- Not caused by sample rate mismatch - the issue is in the model's output waveform itself
- Increased token sampling speed (from ~8 it/s to ~20 it/s) in logs

## Root Cause Analysis

### 1. **Incorrect Positional Embeddings During Generation**

**Location**: `src/chatterbox/models/t3/t3.py` lines ~367-369

**The Problem**:
```python
# WRONG: Position based on iteration counter only
next_token_embed = self.speech_emb(next_token)
next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
```

The positional embedding was calculated using only the loop iteration counter `i`, which represents the number of **new** tokens generated. However:

- The actual sequence includes: conditioning embeddings + text tokens + BOS token + generated tokens
- With KV caching, only the new token is processed, but its position must be relative to the **entire sequence**
- As generation progresses, the positional embeddings become increasingly misaligned

**Why This Causes Acceleration**:
1. **Position Drift**: Misaligned positions confuse the transformer's attention mechanism
2. **Duration Compression**: The model learned during training that certain position ranges correspond to certain duration patterns. Incorrect positions trigger wrong duration predictions
3. **Cumulative Effect**: The error compounds - by token 1000, the position is off by hundreds of positions
4. **Attention Cascade**: Incorrect positional information cascades through all attention layers, causing the model to predict compressed speech frames

### 2. **Over-Aggressive Alignment Stream Analyzer**

**Location**: `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`

The alignment analyzer uses fixed thresholds for detecting hallucinations and repetitions:
- `long_tail` threshold: 5 frames
- `alignment_repetition` threshold: 5 activations

For long sequences (800-1000 tokens), these fixed thresholds become too sensitive and may:
- Force premature EOS tokens
- Misinterpret normal variation as errors
- Apply excessive pressure on the generation process

## Solutions Implemented

### Fix 1: Correct Positional Embeddings (CRITICAL)

**File**: `src/chatterbox/models/t3/t3.py`

**Changes**:
1. Track the initial sequence length (conditioning + text + BOS):
   ```python
   # Track the initial sequence length (conditioning + text + BOS)
   # This is crucial for correct positional embeddings during generation
   initial_seq_len = inputs_embeds.size(1)
   ```

2. Calculate position correctly during generation:
   ```python
   # Get embedding for the new token with CORRECT positional embedding
   # Position should be: initial_seq_len (cond+text+BOS) + current_step
   # This ensures positional embeddings stay aligned throughout generation
   current_position = initial_seq_len + i
   next_token_embed = self.speech_emb(next_token)
   next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(current_position)
   ```

**Impact**: This ensures the model always knows its true position in the sequence, preventing temporal drift and duration compression.

### Fix 2: Adaptive Alignment Thresholds

**File**: `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`

**Changes**:
1. Scale hallucination detection thresholds with sequence length:
   ```python
   # FIXED: Be more lenient with long sequences to prevent premature acceleration
   # Increase threshold proportionally with sequence length to avoid false positives
   long_tail_threshold = min(10, 5 + T // 200)  # Scale threshold: 5 frames base, +1 per 200 frames
   long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= long_tail_threshold)
   
   # FIXED: Also scale this threshold for long sequences
   repetition_threshold = min(10, 5 + T // 200)
   alignment_repetition = self.complete and (A[self.completed_at:, :-5].max(dim=1).values.sum() > repetition_threshold)
   ```

2. More lenient discontinuity checks for long sequences:
   ```python
   # FIXED: More lenient discontinuity check for long sequences
   # Allow larger jumps to account for spaces, punctuation, and model uncertainty
   max_backward_jump = min(4, 2 + T // 200)
   max_forward_jump = min(10, 7 + T // 100)
   discontinuity = not(-max_backward_jump < cur_text_posn - self.text_position < max_forward_jump)
   ```

**Impact**: Prevents the alignment analyzer from being overly aggressive with long sequences, reducing false positives that could force early termination or apply unwanted constraints.

## Testing Recommendations

1. **Short Text (< 100 tokens)**: Should work identically to before
2. **Medium Text (100-500 tokens)**: Should maintain consistent pace without acceleration
3. **Long Text (500-1000 tokens)**: Main test case - should now maintain natural pace throughout
4. **Very Long Text (> 1000 tokens)**: Consider implementing automatic chunking (already available via `auto_split=True`)

## Test Cases

```python
# Test 1: Long paragraph generation
tts.generate(
    text="Your long text here... (500+ words)",
    language_id="en",
    temperature=0.8,
    cfg_weight=0.5,
)

# Test 2: Very long text with auto-split
tts.generate(
    text="Your very long text here... (1000+ words)",
    language_id="en",
    auto_split=True,  # Automatically splits into sentences
)
```

## Expected Results

- **Before Fix**: Speech starts normal, accelerates progressively, becomes very fast/skips words by end
- **After Fix**: Consistent natural pace throughout the entire generation, regardless of length

## Additional Notes

- The `auto_split=True` feature (already implemented in `mtl_tts.py`) provides an alternative approach by chunking long text into sentences
- For extremely long texts (> 2000 tokens), sentence-level chunking is still recommended for best quality
- The positional embedding fix is the most critical change and should resolve the core issue
- The alignment analyzer improvements are complementary and reduce edge case errors

## References

- GitHub Issue #327: "Speedup Voice and Noise in Multilingual Model"
- Related to transformer positional encoding behavior with KV caching
- Similar issues reported in other autoregressive speech models when positional information drifts

---

**Fix applied**: October 23, 2025
**Files modified**:
- `src/chatterbox/models/t3/t3.py`
- `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py`
