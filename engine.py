import numpy as np

# init weights functions

def init_words_weights(vocab_size, word_representor_size):
    word_weights = np.random.randn(vocab_size, word_representor_size) * (1 / np.sqrt(word_representor_size)) # create weight matrix for all input_words
    # more dimension for a word means a higher sum up result, we apply a division to regulate this case
    return word_weights

def init_qkv_weights(word_representor_size):
    W_Q = np.random.randn(word_representor_size, word_representor_size) * (1 / np.sqrt(word_representor_size))
    W_K = np.random.randn(word_representor_size, word_representor_size) * (1 / np.sqrt(word_representor_size))
    W_V = np.random.randn(word_representor_size, word_representor_size) * (1 / np.sqrt(word_representor_size))
    return W_Q, W_K, W_V

def init_ffn_weights(word_representor_size, d_ff):
    W1 = np.random.randn(word_representor_size, d_ff) * (1 / np.sqrt(word_representor_size))
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, word_representor_size) * (1 / np.sqrt(d_ff))
    b2 = np.zeros(word_representor_size)
    return W1, b1, W2, b2

# NEW: output projection weights (decoder hidden state -> vocab logits)
def init_output_weights(word_representor_size, vocab_size):
    W_out = np.random.randn(word_representor_size, vocab_size) * (1 / np.sqrt(word_representor_size))
    b_out = np.zeros(vocab_size)
    return W_out, b_out


# pipeline functions

def word_embedding(word_weights, input_seq, word_activation_func): # input_seq : 00000010000,0010000000................
    word_embedded_seq = []
    for one_hot_vector in input_seq:
        word_embedded_seq.append(word_activation_func(one_hot_vector @ word_weights))
    return np.array(word_embedded_seq)
    # input_seq_len x word_representor_size

    
def positional_encoding(word_embedded_seq):
    seq_len, word_representor_size = word_embedded_seq.shape
    PE = np.zeros((seq_len, word_representor_size))

    # Each dimension i gets a different wave frequency:
    #   low i  -> divisor near 1    -> fast-changing angle -> rapid oscillation across positions
    #   high i -> divisor near 10000 -> slow-changing angle -> near-flat across positions
    # Even/odd dims are paired (0&1, 2&3, ...): both use the same frequency,
    # just phase-shifted 90 degrees (sin vs cos), giving each position a unique signature
    
    for pos in range(seq_len):
        for i in range(word_representor_size):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / word_representor_size)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i - 1) / word_representor_size)))
    return word_embedded_seq + PE
    # input_seq_len x word_representor_size


# to calculate the dot product of the input sequence and the weights, which is used to project the input sequence into a different space (key, query, or value)
def project(seq, weights):
    return np.dot(seq, weights)


# NEW: builds a causal (look-ahead) mask so position i can only attend to positions <= i.
# shape: (seq_len, seq_len), 0 where allowed, -inf where forbidden (added to scores before softmax)
def causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) # upper triangle above diagonal = future positions
    mask = mask * -1e9 # large negative number so softmax pushes these to ~0, not literal -inf (avoids nan on all-masked rows)
    return mask


#we use this func for both self-atteniton and cross-attention ,its why we need weight parameters
def attention(query_input, kv_input, key_w, value_w, query_w, attention_activation_func, mask=None):
    
    # query_input: sequence that Query is generated from (self-attn: = kv_input, cross-attn: decoder's own state)
    # kv_input: sequence that Key/Value are generated from (self-attn: = query_input, cross-attn: encoder_output)
    word_representor_size = query_input.shape[-1]
    
    #values for key, query and value matrices
    keys = project(kv_input, key_w)
    queries = project(query_input, query_w)
    values = project(kv_input, value_w)
    
    #calculate attention scores
    scores = np.dot(queries, keys.T) / np.sqrt(word_representor_size)

    if mask is not None:
        scores = scores + mask
    
    #apply softmax to get attention weights
    weights = attention_activation_func(scores)
    outputs = np.dot(weights, values) 
    cache = {
        'query_input': query_input, 'kv_input': kv_input,
        'key_weights': key_w, 'query_weights': query_w, 'value_weights': value_w,
        'keys': keys, 'queries': queries, 'values': values,
        'weights': weights,
        'outputs': outputs
    }
    
    return outputs, cache



def residual_connection(input_seq, output_seq):
    return input_seq + output_seq


def ffn(residual1, W1, b1, W2, b2, ffn_activation_func):
    hidden = ffn_activation_func(residual1 @ W1 + b1)
    output = hidden @ W2 + b2
    return output



def embed_input(one_hot_seq, word_weights, word_activation_func):
    word_embedded_seq = word_embedding(word_weights, one_hot_seq, word_activation_func)
    pos_encoded_seq = positional_encoding(word_embedded_seq)
    return pos_encoded_seq


def encoder_layer(layer_input, key_w, value_w, query_w, ffn_w1, ffn_b1, ffn_w2, ffn_b2, attention_activation_func, ffn_activation_func):
    
    attention_output, attention_cache = attention(layer_input, layer_input, key_w, value_w, query_w, attention_activation_func)
    residual1 = residual_connection(layer_input, attention_output)

    ffn_output = ffn(residual1, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ffn_activation_func)
    output_seq = residual_connection(residual1, ffn_output)

    layer_cache = {
        'layer_input': layer_input,
        'attention_cache': attention_cache,
        'residual1': residual1,
        'ffn_output': ffn_output,
        'output_seq': output_seq
    }
    return output_seq, layer_cache


# ONE decoder layer: self-attn -> residual -> cross-attn -> residual -> ffn -> residual (3 residuals)
def decoder_layer(layer_input, encoder_output,
                   self_key_w, self_value_w, self_query_w,
                   cross_key_w, cross_value_w, cross_query_w,
                   ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                   attention_activation_func, ffn_activation_func):

    # self-attention: query = key = value = layer_input
    # NEW: causal mask so each position can't see future positions (autoregressive requirement)
    seq_len = layer_input.shape[0]
    self_mask = causal_mask(seq_len)
    self_attn_output, self_attn_cache = attention(layer_input, layer_input, self_key_w, self_value_w, self_query_w, attention_activation_func, mask=self_mask)
    residual1 = residual_connection(layer_input, self_attn_output)

    # cross-attention: query = residual1 (decoder side), key/value = encoder_output
    cross_attn_output, cross_attn_cache = attention(residual1, encoder_output, cross_key_w, cross_value_w, cross_query_w, attention_activation_func)
    residual2 = residual_connection(residual1, cross_attn_output)

    ffn_output = ffn(residual2, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ffn_activation_func)
    output_seq = residual_connection(residual2, ffn_output)

    layer_cache = {
        'layer_input': layer_input,
        'self_attn_cache': self_attn_cache,
        'residual1': residual1,
        'cross_attn_cache': cross_attn_cache,
        'residual2': residual2,
        'ffn_output': ffn_output,
        'output_seq': output_seq
    }
    return output_seq, layer_cache


def encoder(one_hot_seq, word_weights, key_weights, value_weights, query_weights,
            ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            word_activation_func, attention_activation_func, ffn_activation_func, num_layers=1):
    # we may need additional encoder layers to solve more complex problems, so we can add a parameter to specify the number of layers.there is where parallel proccess comes in.

    # embedding happens ONCE, before the layer loop
    output_seq = embed_input(one_hot_seq, word_weights, word_activation_func)

    encoder_cache = []
    # input seq & output_seq size is same for all layers, so we can drive a multi-layer encoder
    for _ in range(num_layers):
        output_seq, layer_cache = encoder_layer(output_seq, key_weights, value_weights, query_weights,
                                                  ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                                                  attention_activation_func, ffn_activation_func)
        encoder_cache.append(layer_cache)
    return output_seq, encoder_cache # cache also contains the output_seq.



def decoder(one_hot_seq, encoder_output, word_weights,
            self_key_weights, self_value_weights, self_query_weights,
            cross_key_weights, cross_value_weights, cross_query_weights,
            ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            word_activation_func, attention_activation_func, ffn_activation_func, num_layers=1):

    # embedding happens ONCE, before the layer loop
    output_seq = embed_input(one_hot_seq, word_weights, word_activation_func)

    decoder_cache = []
    for _ in range(num_layers):
        output_seq, layer_cache = decoder_layer(output_seq, encoder_output,
                                                  self_key_weights, self_value_weights, self_query_weights,
                                                  cross_key_weights, cross_value_weights, cross_query_weights,
                                                  ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                                                  attention_activation_func, ffn_activation_func)
        decoder_cache.append(layer_cache)
    return output_seq, decoder_cache


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True) # subtract max for numerical stability
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def output_projection(decoder_output, W_out, b_out):
    logits = decoder_output @ W_out + b_out
    probs = softmax(logits)
    return probs, logits



def transformer(src_one_hot_seq, tgt_one_hot_seq,
                 src_word_weights, tgt_word_weights,
                 enc_key_w, enc_value_w, enc_query_w,
                 enc_ffn_w1, enc_ffn_b1, enc_ffn_w2, enc_ffn_b2,
                 dec_self_key_w, dec_self_value_w, dec_self_query_w,
                 dec_cross_key_w, dec_cross_value_w, dec_cross_query_w,
                 dec_ffn_w1, dec_ffn_b1, dec_ffn_w2, dec_ffn_b2,
                 W_out, b_out,
                 word_activation_func, attention_activation_func, ffn_activation_func,
                 num_encoder_layers=1, num_decoder_layers=1):

    encoder_output, encoder_cache = encoder(src_one_hot_seq, src_word_weights,
                                             enc_key_w, enc_value_w, enc_query_w,
                                             enc_ffn_w1, enc_ffn_b1, enc_ffn_w2, enc_ffn_b2,
                                             word_activation_func, attention_activation_func, ffn_activation_func,
                                             num_layers=num_encoder_layers)

    decoder_output, decoder_cache = decoder(tgt_one_hot_seq, encoder_output, tgt_word_weights,
                                             dec_self_key_w, dec_self_value_w, dec_self_query_w,
                                             dec_cross_key_w, dec_cross_value_w, dec_cross_query_w,
                                             dec_ffn_w1, dec_ffn_b1, dec_ffn_w2, dec_ffn_b2,
                                             word_activation_func, attention_activation_func, ffn_activation_func,
                                             num_layers=num_decoder_layers)

    probs, logits = output_projection(decoder_output, W_out, b_out)

    cache = {
        'encoder_output': encoder_output,
        'encoder_cache': encoder_cache,
        'decoder_output': decoder_output,
        'decoder_cache': decoder_cache,
        'logits': logits,
        'probs': probs
    }
    return probs, cache