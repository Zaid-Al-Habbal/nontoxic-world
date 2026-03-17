import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StackedBiGRUModel(nn.Module):

    def __init__(
        self,
        vocab_size: int = 30_000,
        embed_dim: int = 256,
        n_layers: int = 2,
        hidden_dim: int = 32,
        pad_id: int = 0,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * hidden_dim, 6)

    def forward(self, encodings: dict) -> torch.Tensor:
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=1)
        packed = pack_padded_sequence(
            embeddings, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _outputs, h_n = self.gru(packed)

        # Concatenate the top forward and backward hidden states
        n_dims = self.output.in_features
        top_states = h_n[-2:].permute(1, 0, 2).reshape(-1, n_dims)
        return self.output(top_states)


class StackedBiGRUWithPretrainedEmbedModel(nn.Module):

    def __init__(
        self,
        pretrained_embeddings: nn.Embedding,
        n_layers: int = 2,
        hidden_dim: int = 32,
        pad_id: int = 0,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = pretrained_embeddings.weight.data
        self.embed = nn.Embedding.from_pretrained(weights, freeze=True)
        embed_dim = weights.shape[-1]  # 768 for bert-base-uncased

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * hidden_dim, 6)

    def forward(self, encodings: dict) -> torch.Tensor:
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=1)
        packed = pack_padded_sequence(
            embeddings, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _outputs, h_n = self.gru(packed)

        n_dims = self.output.in_features
        top_states = h_n[-2:].permute(1, 0, 2).reshape(-1, n_dims)
        return self.output(top_states)


class StackedBiGRUWithScaledAttention(nn.Module):

    def __init__(
        self,
        pretrained_embeddings: nn.Embedding,
        n_layers: int = 2,
        hidden_dim: int = 32,
        pad_id: int = 0,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = pretrained_embeddings.weight.data
        self.embed = nn.Embedding.from_pretrained(weights, freeze=True)
        embed_dim = weights.shape[-1]  # 768

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        gru_output_dim = 2 * hidden_dim
        self.query = nn.Linear(gru_output_dim, gru_output_dim)
        self.key = nn.Linear(gru_output_dim, gru_output_dim)
        self.value = nn.Linear(gru_output_dim, gru_output_dim)

        # Stored as a buffer so it moves with .to(device) automatically
        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(gru_output_dim, dtype=torch.float32))
        )

        self.output = nn.Linear(gru_output_dim, 6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encodings: dict) -> torch.Tensor:
        embeddings = self.embed(encodings["input_ids"])
        lengths = encodings["attention_mask"].sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        packed = pack_padded_sequence(
            embeddings, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.gru(packed)
        gru_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        Q = self.query(gru_outputs)
        K = self.key(gru_outputs)
        V = self.value(gru_outputs)

        # (batch, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Mask padding positions before softmax
        attention_mask = encodings["attention_mask"].unsqueeze(1)
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float("-inf")
        )

        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))

        # (batch, seq_len, gru_output_dim)
        context = torch.matmul(attention_weights, V)

        # Mean-pool over valid (non-padding) positions
        mask_expanded = encodings["attention_mask"].unsqueeze(-1).float()
        context_vector = (context * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        return self.output(context_vector)