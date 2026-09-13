from .attention import multi_head_attention, init_multihead_weights
from .layers import LayerNorm, init_layer_norm_params, ffn, init_ffn_weights


def init_encoder(d_model, d_ff, h, N=6):
    layers = []
    for i in range(N):
        layers.append({
            "self_attn_weights": init_multihead_weights(d_model, h),
            "ln1": init_layer_norm_params(d_model),
            "ffn_weights": init_ffn_weights(d_model, d_ff),
            "ln2": init_layer_norm_params(d_model),
        })
    return layers

# we use diffrent weight set for each layer that contains ALL WEİGHTS
def encoder(x, layers):
    for layer in layers:
        attention_out = multi_head_attention(x, x, x, layer["self_attn_weights"])
        x = LayerNorm(x + attention_out, layer["ln1"]["gamma"], layer["ln1"]["beta"])

        ffn_out = ffn(x, layer["ffn_weights"]["w1"], layer["ffn_weights"]["b1"],
                         layer["ffn_weights"]["w2"], layer["ffn_weights"]["b2"])
        x = LayerNorm(x + ffn_out, layer["ln2"]["gamma"], layer["ln2"]["beta"])
    return x
