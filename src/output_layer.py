import numpy as np
from .attention import softmax


def init_output_layer(d_model, vocab_size):
    W_out = np.random.randn(d_model, vocab_size) * np.sqrt(1.0/d_model)
    # scale by sqrt(1/d_model) to prevent output variance from exploding as d_model grows (Xavier init)
    return W_out

def output_layer(x, W_out):
    logits = x @ W_out
    return softmax(logits) # seq_len x vocab_size
