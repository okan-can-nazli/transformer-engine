import numpy as np
from .attention import softmax


def init_output_layer(d_model, vocab_size):
    W_out = np.random.randn(d_model, vocab_size) * np.sqrt(1.0/d_model)
    # scale by sqrt(1/d_model) to prevent output variance from exploding as d_model grows (Xavier init)
    return W_out

# returns (probs, cache) - cache holds x/logits (needed for backprop through the softmax+linear)
def output_layer(x, W_out):
    logits = x @ W_out
    probs = softmax(logits) # seq_len x vocab_size

    cache = {"x": x, "logits": logits, "probs": probs}
    return probs, cache
