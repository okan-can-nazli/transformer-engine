import numpy as np
from src.embeddings import init_word_embedding_table, build_vocab, get_embeddings, get_PE
from src.encoder import init_encoder, encoder
from src.decoder import init_decoder, decoder
from src.output_layer import init_output_layer, output_layer

d_model, d_ff, h, N = 32, 64, 4, 2

src_sentence = "the cat sat on the mat"
tgt_sentence = "kedi paspasin uzerinde oturdu"

src_vocab = build_vocab(src_sentence)
tgt_vocab = build_vocab(tgt_sentence)

src_emb_table = init_word_embedding_table(len(src_vocab), d_model)
tgt_emb_table = init_word_embedding_table(len(tgt_vocab), d_model)

src_emb = get_embeddings(src_sentence, src_vocab, src_emb_table)
tgt_emb = get_embeddings(tgt_sentence, tgt_vocab, tgt_emb_table)

src_x = src_emb + get_PE(src_emb.shape[0], d_model)
tgt_x = tgt_emb + get_PE(tgt_emb.shape[0], d_model)

enc_layers = init_encoder(d_model, d_ff, h, N)
dec_layers = init_decoder(d_model, d_ff, h, N)

encoder_output = encoder(src_x, enc_layers)
print("encoder_output shape:", encoder_output.shape)

decoder_output = decoder(tgt_x, dec_layers, encoder_output)
print("decoder_output shape:", decoder_output.shape)
print("any NaN in decoder_output?", np.isnan(decoder_output).any())

vocab_size = len(tgt_vocab)
W_out = init_output_layer(d_model, vocab_size)
probs = output_layer(decoder_output, W_out)
print("probs shape:", probs.shape)
print("row sums (should all be ~1.0):", probs.sum(axis=-1))
