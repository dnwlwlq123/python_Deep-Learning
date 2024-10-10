import torch
import torch.nn as nn

from manual import RNNCellManual, LSTMCellManual

class EncoderState:
    def __init__(self, hidden, **kargs):
        self.hidden = hidden
        for k, v in kargs.items():
            exec(f'self.{k} = v')

    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()

class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, model_type, hidden_dim):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)
    def forward(self, source):
        # example = [[
        batch_size, seq_length = source.size()

        #embedded: batch_size, embedding_dim
        embedded = self.embedding(source)
        encoder_state = EncoderState(model_type = self.model_type).initialize()

        # for t in range(seq_length):
        #     x_t = embedded[:, t, :]
        #     if self.model_type == RNNCellManual:
        #         h_t = self.cell(x_t, h_t)
        #     elif self.model_type == LSTMCellManual:
        #         h_t, c_t = self.cell(x_t, h_t, c_t)

        for t in range(seq_length):
            x_t = embedded[:, t, :]
            encoder_state = self.cell(x_t, encoder_state)
        return encoder_state



class Decoder(nn.Module):
    pass

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        encoder_hidden = self.encoder(source)
        outputs = self.decoder(target, encoder_hidden)

        return outputs

if __name__ == '__main__':
    encoder = Encoder(source_vocab, embedding_dim, hidden_dim, RNNCellManual)
    model = Seq2Seq(encoder = ...
                    decoder = ...)
    train, valid, test, parse_file(...)

    for source_batch, target_batch in train:
        model(source_batch)