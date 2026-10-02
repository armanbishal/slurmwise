---
id: cuda_error
title: CUDA runtime or kernel error (not OOM)
kinds: [cuda_error]
tags: [device-side assert, no kernel image, CUDA driver, CUBLAS_STATUS, cuDNN, CUDA_LAUNCH_BLOCKING]
severity: mixed
---
# CUDA runtime or kernel error (not OOM)

## Symptoms
- `CUDA error: device-side assert triggered`, with `Indexing.cu ... srcIndex < srcSelectDimSize` above it
- `no kernel image is available for execution on the device`
- `CUDA driver version is insufficient for CUDA runtime version`
- `CUBLAS_STATUS_NOT_INITIALIZED` or a cuDNN error

## Causes
- Device-side assert: a token id >= vocab size, or a label out of range (tokenizer/model mismatch)
- No kernel image: the torch/CUDA build lacks kernels for this GPU's compute capability
- Driver insufficient: the node's driver is older than the toolkit torch was built against
- CUBLAS/cuDNN init: often a disguised OOM at kernel init, or a bad GPU

## Fixes
1. Device-side assert: check `tokenizer.vocab_size <= model.config.vocab_size`; call
   `model.resize_token_embeddings(len(tokenizer))` after adding tokens. Re-run with
   `CUDA_LAUNCH_BLOCKING=1` for the real line number.
2. No kernel image: load a matching CUDA module, reinstall torch for that CUDA, or pin the
   node type with `--constraint`.
3. Driver insufficient: use a wheel built for an older CUDA, or request newer-driver nodes.
4. CUBLAS/cuDNN: retry on another node with `--exclude=<node>`; if it repeats, report the node.

## Verify
- Re-run with `CUDA_LAUNCH_BLOCKING=1`; check `nvidia-smi` and the loaded CUDA module version.
