import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import RNNCell
from torch.utils.data import DataLoader
from data_handler import Vocabulary
from rnn_cells import RNNCellManual, LSTMCellManual



class EncoderState:
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')

    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()

class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)

    def forward(self, source):
        batch_size, seq_length = source.size()
        embedded = self.embedding(source)
        encoder_state = self.cell.initialize(batch_size)

        for t in range(seq_length):
            x_t = embedded[:, t, :]
            if isinstance(self.cell, RNNCellManual):
                encoder_state = self.cell(x_t, encoder_state)
            elif isinstance(self.cell, LSTMCellManual):
                encoder_state = self.cell(x_t, *encoder_state)

        return encoder_state

class Decoder(nn.Module):
    def __init__(self, target_vocab, embedding_dim, hidden_dim, model_type):
        super(Decoder, self).__init__()
        self.target_vocab = target_vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        # self.output_dim = output_dim

        self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size)

    def forward(self, target, encoder_last_state, teacher_forcing_ratio = 0.5):
        #target: batch_size, seq_length
        batch_size, seq_length = target.size()
        #encoder_last_state: (hidden,)(hidden, hidden)
        input = torch.tensor([self.target_vocab.SOS_IDX for _ in range(batch_size)])

        outputs = []
        decoder_state = encoder_last_state
        for t in range(seq_length):
            embedded = self.embedding(input)
            if isinstance(self.cell, RNNCellManual):
                decoder_state = self.cell(embedded, decoder_state)
                output = self.h2o(decoder_state)
            elif isinstance(self.cell, LSTMCellManual):
                hidden, cell = decoder_state
                hidden, cell = self.cell(embedded, hidden, cell)
                decoder_state = (hidden, cell)
                output = self.h2o(hidden)

            outputs.append(output)

            if random.random() < teacher_forcing_ratio and t < seq_length -1:
                input = target[:, t+1]
            else :
                input = torch.argmax(output, dim = 1)
        #outputs : seq_length, batch_size, output_dim, vocab_size
        return torch.stack(outputs, dim = 1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, source, target):
        encoder_hidden = self.encoder(source) 
        outputs = self.decoder(target, encoder_hidden)

        return outputs

