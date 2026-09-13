import numpy as np


def init_layer_norm_params(d_model): # gamma and beta are also learnable? YES ! kinda
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    return {"gamma" : gamma, "beta" : beta}

# returns (out, cache) - cache holds mean/var/normalized (needed for backprop through the normalization)
def LayerNorm(x, gamma, beta, eps=1e-6):  # x = embedding + pe + scaled_dot_product_attention_output
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * normalized + beta # seq_len x d_model

    cache = {"x": x, "mean": mean, "var": var, "normalized": normalized, "gamma": gamma, "out": out}
    return out, cache

def init_ffn_weights(d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff) * np.sqrt(1.0/d_model)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * np.sqrt(1.0/d_ff)
    b2 = np.zeros(d_model)
    return {"w1": W1, "b1": b1, "w2": W2, "b2": b2}

# returns (out, cache) - cache holds pre/post relu values (needed for backprop through relu)
def ffn(x, w1, b1, w2, b2): # x is output of layernorm seq_len x d_model
    pre_relu = x @ w1 + b1
    post_relu = np.maximum(0, pre_relu) # np.maximum makes negative values into 0
    out = post_relu @ w2 + b2 # seq_len x d_model

    cache = {"x": x, "pre_relu": pre_relu, "post_relu": post_relu, "out": out}
    return out, cache
