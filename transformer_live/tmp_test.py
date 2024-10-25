print(len(en2fr.source_sentences))
print(len(en2fr.target_sentences))

for src, tgt in zip(en2fr.source_sentences[:10], en2fr.target_sentences[:10]):
    print(f"Source: {src}")
    print(f"Target: {tgt}")

print(en2fr.source_vocab.pad_idx)
print(en2fr.target_vocab.pad_idx)

print(en2fr.source_vocab.vocab_size)
print(en2fr.target_vocab.vocab_size)

for idx, (batch_x, batch_y) in enumerate(en2fr.data):
    print(batch_x.shape)
    print(batch_y.shape)

    print(batch_x[:1])
    print(batch_y[:1])

    if idx > 1:
        break