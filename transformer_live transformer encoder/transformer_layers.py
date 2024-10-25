import torch 
import torch.nn as nn 
from typing import List

from functorch.dim.reference import positional

class LayerNormalization(nn.Module):
    def __init__(self,
                 input_dim,
                 eps: float = 1e-6,
                 ):
        self.gamma = nn.Parameter(torch.zeros(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(self,
                x: torch.tensor
                ) -> torch.tensor:
        mean = torch.mean(x)
        std = torch.std(x)

        return self.gamma * (x-mean) / (std + self.eps) + self.beta

class PositionalEncoding(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 max_length: int
                 ):
        super(PositionalEncoding, self).__init__()
        positional_encoding = torch.zeros(max_length, embedding_dim)

        for p in range(position):
            for i in range(embedding_dim // 2):
                arg = (position / 10000) ** (i / embedding_dim)
                positional_encoding[p][i] = torch.sin(arg)
                if 2*i + 1< embedding_dim:
                    positional_encoding[p][i+1] = torch.cos(arg)


        position = torch.arange(0, max_length).float() # position: (max-len, )
        i = torch.arange(0, embedding_dim, 2).float()
        arg = (position / 10000) ** (i/embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(arg)
        positional_encoding[:, 1::2] = torch.cos(arg)

        self.positional_encoding = positional_encoding
    def forward(self,
                x: torch.tensor
                ) -> torch.tensor:
        x = x + self.positional_encoding
        return x

class SelfAttention(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        attention_head_dim: int, 
    ):
        self.W_q: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)
        self.W_k: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)
        self.W_v: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)

        self.softmax: nn.SoftMax = nn.SoftMax(dim = -1)
        self.attention_head_dim = attention_head_dim

    def forward(
        self, 
        x: torch.tensor
    ):
        Q: torch.tensor = self.W_q(x) 
        K: torch.tensor = self.W_k(x) 
        V: torch.tensor = self.W_v(x) 

        score: torch.tensor = Q @ K.transpose(-2, -1) / self.attention_head_dim ** 0.5

        attention_distribution: torch.tensor = self.softmax(score) 
        Z: torch.tensor = attention_distribution @ V 

        return Z 


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        num_heads: int, 
        attention_head_dim: int, 
        # batch_first: bool = True, 
    ):
        super(MultiheadSelfAttention, self).__init__()
        self.heads: List[SelfAttention] = [SelfAttention(embedding_dim, attention_head_dim) for _ in range(num_heads)]
        self.layer: nn.Linear = nn.Linear(num_heads * attention_head_dim, embedding_dim)

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        x = torch.cat([head(x) for head in self.head], dim = 1)
        x = self.layer(x) 

        return x 

        

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        num_heads: int, 
        attention_head_dim: int, 
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention: MultiheadSelfAttention = MultiheadSelfAttention(embedding_dim, num_heads, attention_head_dim)
        self.ff: nn.Linear = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm: LayerNormalization = LayerNormalization(embedding_dim)

    def forward(
        self, 
        x: torch.tensor, 
    ) -> torch.tensor:
        after = self.self_attention(x)
        self.layer_norm(x + after)
        x = self.ff(x) 
        return x 

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        embedding_dim: int, 
        num_heads: int, 
        attention_head_dim: int, 
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers: List[TransformerEncoderLayer] = [TransformerEncoderLayer(
            embedding_dim, num_heads, attention_head_dim) for _ in range(num_layers)]
        self.positional_encoding: PositionalEncoding = PositionalEncoding(embedding_dim, )
        

    def forward(
        self, 
        x: torch.tensor, 
    ) -> torch.tensor:
        for layer in self.layers:
            x = layer(x) 
        return x 

class Transformer(nn.Module):
    def __init__(
        self, 
        num_heads: int, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        embedding_dim: int, 
        num_layers: int, 
        attention_head_dim: int, 
    ):
        super(Transformer, self).__init__() 
        self.encoder_embedding: nn.Embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.decoder_embedding: nn.Embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.encoder: TransformerEncoder =  TransformerEncoder(num_layers, embedding_dim, num_heads, attention_head_dim)
        self.decoder: TransformerDecoder = TransformerDecoder(num_layers, embedding_dim, num_heads)
        self.final_layer = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(
        self,
        src: torch.tensor, 
        tgt: torch.tensor, 
    ) -> torch.tensor:
        src_embedding: torch.tensor = self.encoder_embedding(src)
        encoder_output: torch.tensor = self.encoder(src_embedding)

        tgt_embedding: torch.tensor = self.decoder_embedding(tgt)
        decoder_output: torch.tensor = self.decoder(tgt_embedding, encoder_output)

        out: torch.tensor = self.final_layer(decoder_output)

        return out