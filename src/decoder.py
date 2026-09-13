from .attention import multi_head_attention, init_multihead_weights, get_causal_mask
from .layers import LayerNorm, init_layer_norm_params, ffn, init_ffn_weights


def init_decoder(d_model, d_ff, h, N=6):
    layers = []
    for i in range(N):
        layers.append({
            "self_attn_weights": init_multihead_weights(d_model, h),
            "ln1": init_layer_norm_params(d_model),
            "cross_attn_weights": init_multihead_weights(d_model, h),
            "ln2": init_layer_norm_params(d_model),
            "ffn_weights": init_ffn_weights(d_model, d_ff),
            "ln3": init_layer_norm_params(d_model)
        })
    return layers


def decoder(x, layers, encoder_output):
    seq_len = x.shape[0]
    causal_mask = get_causal_mask(seq_len)


    for layer in layers:
        self_attn_out = multi_head_attention(x, x, x, layer["self_attn_weights"], mask=causal_mask)
        x = LayerNorm(x + self_attn_out, layer["ln1"]["gamma"], layer["ln1"]["beta"])

        cross_attn_out = multi_head_attention(q_input=x, k_input=encoder_output, v_input=encoder_output, mha_weights=layer["cross_attn_weights"])
        x = LayerNorm(x + cross_attn_out, layer["ln2"]["gamma"], layer["ln2"]["beta"])

        ffn_out = ffn(x, layer["ffn_weights"]["w1"], layer["ffn_weights"]["b1"],
                         layer["ffn_weights"]["w2"], layer["ffn_weights"]["b2"])
        x = LayerNorm(x + ffn_out, layer["ln3"]["gamma"], layer["ln3"]["beta"])
    return x
