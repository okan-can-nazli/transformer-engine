import numpy as np


def init_layer_norm_params(d_model): # gamma and beta are also learnable?YES
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    return {"gamma" : gamma, "beta" : beta}

def LayerNorm(x, gamma, beta, eps=1e-6):  # x = embedding + pe + scaled_dot_product_attention_output
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta # seq_len x d_model

def init_ffn_weights(d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff) * np.sqrt(1.0/d_model)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * np.sqrt(1.0/d_ff)
    b2 = np.zeros(d_model)
    return {"w1": W1, "b1": b1, "w2": W2, "b2": b2}

def ffn(x, w1, b1, w2, b2): # x is output of layernorm seq_len x d_model
    return np.maximum(0, x @ w1 + b1) @ w2 + b2 # seq_len x d_model
# np.maximum makes negative values into 0
