---
id: cuda_oom
title: GPU out of memory
kinds: [cuda_oom]
tags: [cuda, gpu, torch.OutOfMemoryError, CUDA out of memory, batch size, gradient checkpointing, PYTORCH_CUDA_ALLOC_CONF]
severity: user
---
# GPU out of memory

## Symptoms
- `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB`
- sacct state `FAILED`, exit code `1:0`
- Often at the first forward/backward pass or the first generation step

## Causes
- Batch size or sequence length too large for the card
- For PPO/GRPO-style RL: `num_generations * max_completion_length` dominates the peak;
  generation, not the backward pass, is usually the largest allocation
- fp32 weights where bf16 would do; no gradient checkpointing; full fine-tune instead of LoRA

## Fixes
1. Halve `per_device_train_batch_size`, double `gradient_accumulation_steps`.
2. Cut `max_seq_length` / `max_completion_length` / `max_prompt_length`.
3. Turn on gradient checkpointing.
4. Use bf16 and 4-bit base (QLoRA) with LoRA adapters.
5. Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fight fragmentation.
6. Request a larger card via `--gres` or `--constraint`.

## Verify
- `nvidia-smi` during a short run to watch peak memory, or re-run with the allocator hint.
