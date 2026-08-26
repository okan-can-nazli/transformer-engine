
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


# we use this func for both self-atteniton and cross-attention ,its why we need weight parameters
def attention(input_seq, key_weights, value_weights, query_weights, attention_activation_func):
    
    # word_embedded_seq shape: (seq_len, word_representor_size)
    seq_len, word_representor_size = input_seq.shape
    
    #weights for key, query and value matrices, initialized with random values
    # key_weights = np.random.randn(seq_len, word_representor_size) * (1 / np.sqrt(word_representor_size))
    # query_weights = np.random.randn(seq_len, word_representor_size) * (1 / np.sqrt(word_representor_size))
    # value_weights = np.random.randn(seq_len, word_representor_size) * (1 / np.sqrt(word_representor_size))
    
    #values for key, query and value matrices
    keys = project(input_seq, key_weights)
    queries = project(input_seq, query_weights)
    values = project(input_seq, value_weights)
    
    #calculate attention scores
    attention_scores = np.dot(queries, keys.T) / np.sqrt(word_representor_size)
    
    #apply softmax to get attention weights
    attention_weights = attention_activation_func(attention_scores)
    attention_outputs = np.dot(attention_weights, values),
    attention_cache = {
        'input_seq': input_seq,
        'key_weights': key_weights, 'query_weights': query_weights, 'value_weights': value_weights,
        'keys': keys, 'queries': queries, 'values': values,
        'attention_weights': attention_weights,
        'attention_outputs': attention_outputs
    }
    
    return attention_outputs, attention_cache
    # return attention_output, attention_cache



def residual_connection(input_seq, output_seq):
    return input_seq + output_seq


    
def pipe_layer(one_hot_vector, word_weights, key_weights, value_weights, query_weights, word_activation_func , attention_activation_func):
    
    # Step 1: Word Embedding
    word_embedded_seq = word_embedding(word_weights, one_hot_vector, word_activation_func)
    
    # Step 2: Positional Encoding
    pos_encoded_seq = positional_encoding(word_embedded_seq.shape[1], word_embedded_seq)
    
    # Step 3: Self-Attention
    attention_output, attention_cache = attention(pos_encoded_seq, key_weights, value_weights, query_weights, attention_activation_func)
    
    # Step 4: Residual Connection
    output_seq = residual_connection(pos_encoded_seq, attention_output)
    
    pipe_cache = {
        'word_embedded_weights': word_weights,
        'pos_encoded_seq': pos_encoded_seq,
        'attention_cache': attention_cache,
        'output_seq': output_seq
    }
    return output_seq , pipe_cache



def encoder(input_seq, key_weights, value_weights, query_weights, word_activation_func, attention_activation_func, num_layers=1): # we may neeed additional encoder layers to solve more complex problems, so we can add a parameter to specify the number of layers.there is where parallel proccess comes in.
    output_seq = input_seq
    encoder_cache = []
    # input seq & output_seq size is same for all layers, so we can dive a multi-layer encoder
    for _ in range(num_layers):
        output_seq, pipe_cache = pipe_layer(output_seq, key_weights, value_weights, query_weights, word_activation_func, attention_activation_func)
        encoder_cache.append(pipe_cache)
    return output_seq, encoder_cache # cache also contains the output_seq.



def decoder(input_seq, encoder_output, key_weights, value_weights, query_weights, word_activation_func, attention_activation_func, num_layers=1):
    output_seq = input_seq
    decoder_cache = []
    for _ in range(num_layers):
        output_seq , pipe_cache = pipe_layer(output_seq, key_weights, value_weights, query_weights, word_activation_func, attention_activation_func)
        decoder_cache.append(pipe_cache)
        cross_attention_output, cross_attention_cache = attention(output_seq, encoder_output, attention_activation_func)
        output_seq = residual_connection(output_seq, cross_attention_output)
    return output_seq, decoder_cache    

