import torch
import torch.nn as nn

class RecurrentClassifier(nn.Module):
    """Architecture matching the recurrent models exported by the training notebook."""

    def __init__(self, kind, vocab_size, emb_dim=100, hidden_dim=128, num_classes=6, dropout=0.30):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = nn.LSTM if kind in ["LSTM", "BiLSTM_Attention"] else nn.GRU
        bidirectional = kind == "BiLSTM_Attention"
        self.rnn = rnn_cls(
            emb_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Linear(out_dim, 1) if bidirectional else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, lengths, return_attention=False):
        x = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]

        attention_weights = None
        if self.attention is not None:
            scores = self.attention(output).squeeze(-1).masked_fill(~mask, -1e9)
            attention_weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        else:
            pooled = output[
                torch.arange(output.size(0), device=output.device), lengths - 1
            ]

        logits = self.classifier(self.dropout(pooled))
        if return_attention:
            return logits, attention_weights
        return logits
