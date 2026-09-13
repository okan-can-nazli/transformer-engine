import numpy as np
from src.embeddings import init_word_embedding_table, build_vocab, get_embeddings, get_PE
from src.encoder import init_encoder, encoder
from src.decoder import init_decoder, decoder
from src.output_layer import init_output_layer, output_layer

d_model, d_ff, h, N = 32, 64, 4, 2

src_sentence = "the cat sat on the mat"
tgt_sentence = "kedi paspasin uzerinde oturdu"

print("=" * 60)
print("TRANSFORMER FROM SCRATCH - end-to-end forward pass")
print("=" * 60)
print(f"Source sentence (EN): \"{src_sentence}\"")
print(f"Target sentence (TR): \"{tgt_sentence}\"")
print(f"Hyperparameters: d_model={d_model}, d_ff={d_ff}, heads={h}, layers={N}")

src_vocab = build_vocab(src_sentence)
tgt_vocab = build_vocab(tgt_sentence)

print(f"\nSource vocab ({len(src_vocab)} words): {src_vocab}")
print(f"Target vocab ({len(tgt_vocab)} words): {tgt_vocab}")

src_emb_table = init_word_embedding_table(len(src_vocab), d_model)
tgt_emb_table = init_word_embedding_table(len(tgt_vocab), d_model)

src_emb = get_embeddings(src_sentence, src_vocab, src_emb_table)
tgt_emb = get_embeddings(tgt_sentence, tgt_vocab, tgt_emb_table)

print(f"\nSource embeddings shape (seq_len x d_model): {src_emb.shape}")
print(f"Target embeddings shape (seq_len x d_model): {tgt_emb.shape}")

src_x = src_emb + get_PE(src_emb.shape[0], d_model)
tgt_x = tgt_emb + get_PE(tgt_emb.shape[0], d_model)

enc_layers = init_encoder(d_model, d_ff, h, N)
dec_layers = init_decoder(d_model, d_ff, h, N)

print("\n--- Running encoder ---")
encoder_output, enc_caches = encoder(src_x, enc_layers)
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Encoder produced {len(enc_caches)} layer caches (one dict of intermediates per layer)")

print("\n--- Running decoder (masked self-attn + cross-attn to encoder output) ---")
decoder_output, dec_caches = decoder(tgt_x, dec_layers, encoder_output)
print(f"Decoder output shape: {decoder_output.shape}")
print(f"Decoder produced {len(dec_caches)} layer caches")
print(f"Any NaN in decoder output? {np.isnan(decoder_output).any()}")

print("\n--- Final output layer (logits + softmax over target vocab) ---")
vocab_size = len(tgt_vocab)
W_out = init_output_layer(d_model, vocab_size)
probs, out_cache = output_layer(decoder_output, W_out)
print(f"Output probabilities shape (seq_len x vocab_size): {probs.shape}")
print(f"Row sums (must all be ~1.0 for valid probability distributions): {probs.sum(axis=-1)}")

idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
predicted_ids = np.argmax(probs, axis=-1)
predicted_words = [idx_to_word[i] for i in predicted_ids]
print(f"\nArgmax prediction per position (untrained, so this is meaningless noise -\n"
      f"just confirms the pipeline runs end to end): {predicted_words}")

print("\n--- Cache contents (ready for backprop later) ---")
print(f"Layer-0 encoder self-attn cache keys: {list(enc_caches[0]['attn'].keys())}")
print(f"Layer-0 decoder cache keys: {list(dec_caches[0].keys())}")
print(f"Output layer cache keys: {list(out_cache.keys())}")

print("\n" + "=" * 60)
print("Forward pass complete: embeddings -> encoder -> decoder -> output layer")
print("=" * 60)
