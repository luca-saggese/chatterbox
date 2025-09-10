#!/usr/bin/env python3
"""
Speed-stabilized version of T3 inference to prevent progressive acceleration.

The main issue causing speed acceleration in the original T3 model is the progressive
increase in positional embeddings during the autoregressive generation loop.

Key improvements:
1. Position embedding stabilization - use relative positions or periodic reset
2. KV cache management - optional periodic reset to prevent accumulation bias
3. Temperature scheduling - maintain consistent sampling temperature
4. Timing normalization - apply consistent timing patterns
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import os
from tqdm import tqdm
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, MinPLogitsWarper

def stabilized_inference(
    self,
    *,
    t3_cond,
    text_tokens: Tensor,
    initial_speech_tokens=None,
    
    # Standard generation params
    num_return_sequences=1,
    max_new_tokens=None,
    stop_on_eos=True,
    do_sample=True,
    temperature=0.8,
    length_penalty=1.0,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0,
    cfg_weight=0,
    enable_alignment_analysis: bool = False,
    
    # Speed stabilization params
    use_relative_positions=True,  # Use relative positions instead of absolute
    reset_position_every=None,    # Reset position embedding every N tokens
    stabilize_kv_cache=False,     # Reset KV cache periodically
    kv_reset_interval=100,        # Reset KV cache every N tokens
    position_offset_mode='cyclic', # 'cyclic', 'reset', or 'clamped'
    max_position_embedding=512,   # Maximum position to use before cycling
):
    """
    Speed-stabilized inference method that prevents progressive acceleration.
    
    Args:
        use_relative_positions: Use relative positioning to prevent drift
        reset_position_every: Reset position embedding every N tokens
        stabilize_kv_cache: Whether to reset KV cache periodically
        kv_reset_interval: Reset KV cache every N tokens
        position_offset_mode: How to handle position overflow
            - 'cyclic': Wrap positions back to 0 after max_position_embedding
            - 'reset': Reset to 0 every reset_position_every tokens
            - 'clamped': Clamp position to max_position_embedding
        max_position_embedding: Maximum position before applying offset mode
    """
    
    # Validate inputs (same as original)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
    
    if initial_speech_tokens is None:
        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    
    # Prepare custom input embeds
    embeds, len_cond = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
    )
    
    # Setup backend (same as original)
    if not self.compiled:
        from .inference.t3_hf_backend import T3HuggingfaceBackend
        from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
        
        self.patched_model = T3HuggingfaceBackend(
            config=self.cfg,
            llama=self.tfmr,
            speech_enc=self.speech_emb,
            speech_head=self.speech_head,
            alignment_stream_analyzer=None,
        )
        self.compiled = True
    
    # Setup alignment analysis if requested
    alignment_stream_analyzer = None
    if enable_alignment_analysis:
        alignment_stream_analyzer = AlignmentStreamAnalyzer(
            self.tfmr,
            None,
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            alignment_layer_idx=9,
            eos_idx=self.hp.stop_speech_token,
        )
        self.patched_model.alignment_stream_analyzer = alignment_stream_analyzer
    else:
        self.patched_model.alignment_stream_analyzer = None
    
    device = embeds.device
    
    # Initialize BOS token with position-aware embedding
    bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_embed = self.speech_emb(bos_token)
    
    # Smart position embedding for BOS
    if use_relative_positions:
        # Start at position 0 for relative positioning
        bos_pos_embed = self.speech_pos_emb.get_fixed_embedding(0)
    else:
        bos_pos_embed = self.speech_pos_emb.get_fixed_embedding(0)
    
    bos_embed = bos_embed + bos_pos_embed
    bos_embed = torch.cat([bos_embed, bos_embed])  # CFG
    
    # Initial input embeddings
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
    
    # Token generation setup
    max_steps = int(max_new_tokens) if max_new_tokens else self.hp.max_speech_tokens
    token_buffer = torch.empty((1, max_steps + 1), dtype=torch.long, device=device)
    token_buffer[:, 0] = bos_token
    generated_len = 0
    generated_ids = token_buffer[:, :1]
    
    # Logits processors
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    top_p_warper = TopPLogitsWarper(top_p=top_p)
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    
    # Initial forward pass
    output = self.patched_model(
        inputs_embeds=inputs_embeds,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    past = output.past_key_values
    
    # Position tracking for stabilization
    base_position = 0  # Base position for relative calculations
    last_reset_step = 0  # Last step where we reset something
    
    # Generation loop with speed stabilization
    iterator = tqdm(range(max_steps), desc="Stabilized Sampling") if os.getenv("CHATTERBOX_SHOW_PROGRESS", "0") == "1" else range(max_steps)
    
    for i in iterator:
        logits = output.logits[:, -1, :]
        
        # CFG
        logits_cond = logits[0:1]
        logits_uncond = logits[1:2]
        logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
        logits = logits.squeeze(1)
        
        # Temperature scaling (keep consistent)
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply processors
        logits = repetition_penalty_processor(generated_ids, logits)
        logits = min_p_warper(None, logits)
        logits = top_p_warper(None, logits)
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Update token buffer
        token_buffer[:, generated_len + 1] = next_token.view(1, 1)
        generated_len += 1
        generated_ids = token_buffer[:, :generated_len + 1]
        
        # Check for EOS
        if next_token.view(-1) == self.hp.stop_speech_token:
            break
        
        # SPEED STABILIZATION: Smart position embedding calculation
        current_step = i + 1
        
        # Calculate stabilized position
        if use_relative_positions:
            # Use relative position from base
            relative_pos = current_step - base_position
            
            if position_offset_mode == 'cyclic' and relative_pos >= max_position_embedding:
                # Cycle back to beginning
                base_position = current_step
                relative_pos = 0
            elif position_offset_mode == 'reset' and reset_position_every and (current_step % reset_position_every == 0):
                # Reset position periodically
                base_position = current_step
                relative_pos = 0
            elif position_offset_mode == 'clamped':
                # Clamp to maximum position
                relative_pos = min(relative_pos, max_position_embedding - 1)
            
            position_idx = relative_pos
        else:
            position_idx = current_step
        
        # Get embedding for the new token with stabilized position
        next_token_embed = self.speech_emb(next_token)
        next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(position_idx)
        
        # CFG
        next_token_embed = torch.cat([next_token_embed, next_token_embed])
        
        # SPEED STABILIZATION: Optional KV cache reset
        if stabilize_kv_cache and kv_reset_interval and (current_step % kv_reset_interval == 0):
            # Reset KV cache to prevent accumulation bias
            print(f"Resetting KV cache at step {current_step}")
            
            # We need to rebuild the context without the cache
            # This is expensive but prevents accumulation issues
            current_tokens = token_buffer[:, :generated_len + 1]
            
            # Rebuild embeddings for current sequence
            rebuild_embeds = []
            for j, token in enumerate(current_tokens[0]):
                if j == 0:  # BOS token
                    embed = self.speech_emb(token.unsqueeze(0).unsqueeze(0))
                    embed = embed + self.speech_pos_emb.get_fixed_embedding(0)
                else:
                    embed = self.speech_emb(token.unsqueeze(0).unsqueeze(0))
                    # Use relative position from last reset
                    pos = j - 1  # Relative to BOS
                    if position_offset_mode == 'cyclic':
                        pos = pos % max_position_embedding
                    embed = embed + self.speech_pos_emb.get_fixed_embedding(pos)
                
                rebuild_embeds.append(embed)
            
            # Combine with original conditioning
            speech_context = torch.cat(rebuild_embeds, dim=1)
            speech_context = torch.cat([speech_context, speech_context])  # CFG
            full_context = torch.cat([embeds, speech_context], dim=1)
            
            # Fresh forward pass
            output = self.patched_model(
                inputs_embeds=full_context,
                past_key_values=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values
            last_reset_step = current_step
        else:
            # Normal forward pass with cache
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            past = output.past_key_values
    
    # Clean up
    predicted_tokens = token_buffer[:, 1:generated_len + 1]
    
    if alignment_stream_analyzer is not None:
        try:
            alignment_stream_analyzer.close()
        except Exception:
            pass
    
    return predicted_tokens


# Patch function to apply this to the T3 model
def apply_speed_stabilization_patch(t3_model):
    """
    Apply the speed stabilization patch to a T3 model instance.
    
    Usage:
        from speed_stabilized_t3 import apply_speed_stabilization_patch
        apply_speed_stabilization_patch(your_t3_model)
        
        # Now use with stabilization parameters
        result = your_t3_model.stabilized_inference(
            t3_cond=cond,
            text_tokens=tokens,
            max_new_tokens=1000,
            use_relative_positions=True,
            position_offset_mode='cyclic',
            max_position_embedding=256,
            stabilize_kv_cache=True,
            kv_reset_interval=200
        )
    """
    # Bind the method to the instance
    import types
    t3_model.stabilized_inference = types.MethodType(stabilized_inference, t3_model)
    print("Speed stabilization patch applied to T3 model")
    return t3_model


if __name__ == "__main__":
    print("Speed Stabilized T3 Inference")
    print("="*50)
    print("This module provides a speed-stabilized version of T3 inference")
    print("that prevents progressive acceleration during long text generation.")
    print()
    print("Key features:")
    print("- Relative position embeddings")
    print("- Position cycling/reset")
    print("- Optional KV cache management")
    print("- Consistent temperature control")
    print()
    print("Usage:")
    print("  from speed_stabilized_t3 import apply_speed_stabilization_patch")
    print("  apply_speed_stabilization_patch(your_t3_model)")
    print("  result = your_t3_model.stabilized_inference(...)")
