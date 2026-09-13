import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# main brain of the attention formula
# self-attention phase in my notes but its better to name it based on what it does not where its used
# returns (out, cache) - cache holds everything needed later for backprop
def scaled_dot_product_attention(q, k, v, mask=None):

    # we may indeed use k or v
    d_k = q.shape[-1] # q.shape = (seq_len_num, d_k) [-1] means last element of the tuple
    scores = q @ k.T / np.sqrt(d_k) # seq_len x seq_len (logicts)
    #! q @ k.t means similartiy rate ,after that we apply v to figure out what happens if we give importance to the operation



    # provides mask amoung words
    #         w0    w1    w2
    # w0  [  2.1,  0.5,  1.3 ]
    # w1  [  0.8,  3.0,  1.1 ]
    # w2  [  1.5,  0.9,  2.4 ]

    #in this case w0 make a relation with w2 however a word shouldnt know a word that comes after that
    #         w0      w1      w2
    # w0  [  2.1,   -inf,   -inf  ]
    # w1  [  0.8,    3.0,   -inf  ]
    # w2  [  1.5,    0.9,    2.4  ]
    if mask is not None:
        scores += mask

    probs = softmax(scores) #softmax(scores) : probabilty [0,1]
    out = probs @ v
    # apply transpose to provide inner match in dot product operation

    cache = {"q": q, "k": k, "v": v, "scores": scores, "probs": probs, "out": out}
    return out, cache

def init_multihead_weights(d_model, h): # h : number of heads
    d_k = d_model // h
    heads = []
    for i in range(h):
        scale = np.sqrt(1.0 / d_model)
        heads.append({  # x : seq_len x d_model  @  d_model x d_k = seq_len x d_K : a head
                      # we obtain Q, K, V first with x input then
            "W_Q": np.random.randn(d_model, d_k) * scale,
            "W_K": np.random.randn(d_model, d_k) * scale,
            "W_V": np.random.randn(d_model, d_k) * scale,
        })
    W_O = np.random.randn(d_model, d_model) * np.sqrt(1.0 / d_model) # provides to harmonaize independent head outputs
    return {"heads": heads, "W_O": W_O}


# provides self-attention operation in a parallel small way
# returns (out, cache) - cache holds inputs, per-head caches, and the concat (needed for backprop)
def multi_head_attention(q_input, k_input, v_input, mha_weights, mask=None): # we split the input param x as k,v,q input param for cross validation on decoder
    head_outputs = []
    head_caches = []

    for head in mha_weights["heads"]:
        # matrix smaller
        q = q_input @ head["W_Q"]
        k = k_input @ head["W_K"]
        v = v_input @ head["W_V"]

        head_out, head_cache = scaled_dot_product_attention(q, k, v, mask)  # seq_len x d_k!!!!
        head_outputs.append(head_out)
        head_caches.append(head_cache)

    concat = np.concatenate(head_outputs, axis=-1)  # seq_len x d_model : put all small matrixes together
    out = concat @ mha_weights["W_O"]  # seq_len x d_model : use another weight to dont make them fully independent, create a relation

    cache = {
        "q_input": q_input, "k_input": k_input, "v_input": v_input,
        "head_caches": head_caches,  # one dict per head (q,k,v,scores,probs,out)
        "concat": concat, "out": out,
    }
    return out, cache


def get_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask
    # provides mask amoung words
    #         w0    w1    w2
    # w0  [  2.1,  0.5,  1.3 ]
    # w1  [  0.8,  3.0,  1.1 ]
    # w2  [  1.5,  0.9,  2.4 ]

    #in this case w0 make a relation with w2 however a word shouldnt know a word that comes after that
    #         w0      w1      w2
    # w0  [  2.1,   -inf,   -inf  ]
    # w1  [  0.8,    3.0,   -inf  ]
    # w2  [  1.5,    0.9,    2.4  ]
