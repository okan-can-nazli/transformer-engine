# transformer-engine

A Transformer implemented from scratch in NumPy — no ML frameworks, no autograd. Built by working through "Attention Is All You Need" piece by piece: embeddings, positional encoding, multi-head attention, masking, the encoder stack, and the decoder stack (masked self-attention + cross-attention + feed-forward).

This is a forward-pass implementation for learning purposes — weights are randomly initialized, there is currently no backpropagation or training loop.

## Structure

```
src/
├── embeddings.py     # vocab building, word embeddings, positional encoding
├── attention.py       # scaled dot-product attention, multi-head attention, causal masking
├── layers.py           # LayerNorm, feed-forward network
├── encoder.py           # encoder stack (self-attention + FFN, per layer)
├── decoder.py           # decoder stack (masked self-attention + cross-attention + FFN, per layer)
└── output_layer.py     # final linear projection + softmax over vocab
test.py                  # end-to-end run: sentence -> encoder -> decoder -> output probabilities
```

## Running

```bash
python3 test.py
```

This builds a vocab for a source and target sentence, runs them through the full encoder-decoder pipeline, and checks that the output is a valid probability distribution over the target vocabulary.

## What's implemented

- Word embeddings + sinusoidal positional encoding
- Scaled dot-product attention with optional masking
- Multi-head attention (parallel heads, each with its own Q/K/V projection, recombined via a shared output projection)
- Causal (look-ahead) masking for the decoder
- Cross-attention (decoder queries attending to encoder output)
- Residual connections + LayerNorm around every sub-layer
- Position-wise feed-forward network
- Final linear + softmax output layer
- Forward-pass caching: every layer function returns `(output, cache)`, where `cache` holds every intermediate value needed for backprop (Q/K/V, attention scores/probabilities, LayerNorm mean/var, pre/post-ReLU activations, residual inputs)

## Not yet implemented

- Backpropagation (the cache values are there, the gradient math is not yet)
- Optimizer / training loop
- Batching (currently single-sequence only)

