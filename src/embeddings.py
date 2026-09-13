import numpy as np

#Build VOCAB
def init_word_embedding_table(vocab_size, d_model): # d_model represents each word representor num number(per identity number size)
    return np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)

def build_vocab(sentence):
    words = sentence.split()
    unique_words = set(words)
    return {word: idx for idx, word in enumerate(unique_words)}



# word emebdding
def get_embeddings(sentence, vocab, embedding_table):
    words = sentence.split()
    token_ids = [vocab[word] for word in words] # token ids : [0, 1, 6, 76, 32, 76] seqeunce word order
    embeddings = embedding_table[token_ids] # collect d_models of the seq from table
    return embeddings # seq_len x d_model


# positional encoding
def get_PE(seq_len, d_model):

    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))

            if i % 2 == 0:
                pe[pos][i] = np.sin(angle)
            else:
                pe[pos][i] = np.cos(angle)

    return pe # seq_len x d_model
