"""CNN, BiLSTM + masked attention, and a hybrid of both branches."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentNet(nn.Module):
    def __init__(self, architecture="hybrid", pretrained=True, vocab_size=30522,
                 feature_dim=768, hidden=64, projection=96, dropout=0.35):
        super().__init__()
        self.architecture = architecture
        self.pretrained = pretrained
        self.embedding = (nn.Linear(feature_dim, projection) if pretrained else
                          nn.Embedding(vocab_size, projection, padding_idx=0))
        self.input_dropout = nn.Dropout(dropout)
        out_dim = 0
        if architecture in {"cnn", "hybrid"}:
            self.convs = nn.ModuleList([nn.Conv1d(projection, hidden, k, padding=k//2)
                                        for k in (3, 5, 7)])
            out_dim += hidden * 3
        if architecture in {"bilstm", "hybrid"}:
            self.lstm = nn.LSTM(projection, hidden, batch_first=True, bidirectional=True)
            self.attention = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Tanh(),
                                           nn.Linear(hidden, 1, bias=False))
            out_dim += hidden * 2
        if not out_dim:
            raise ValueError(architecture)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 3))

    def forward(self, inputs, mask):
        mask = mask.bool()
        features = self.embedding(inputs.float() if self.pretrained else inputs.long())
        features = self.input_dropout(features) * mask.unsqueeze(-1)
        branches = []
        if self.architecture in {"cnn", "hybrid"}:
            cnn = features.transpose(1, 2)
            branches += [torch.relu(conv(cnn)).masked_fill(~mask.unsqueeze(1), -1e4)
                         .amax(dim=2) for conv in self.convs]
        if self.architecture in {"bilstm", "hybrid"}:
            packed = pack_padded_sequence(features, mask.sum(1).cpu(),
                                           batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            seq, _ = pad_packed_sequence(packed_out, batch_first=True,
                                          total_length=features.size(1))
            scores = self.attention(seq).squeeze(-1).masked_fill(~mask, -1e4)
            weights = torch.softmax(scores, dim=1)
            branches.append((seq * weights.unsqueeze(-1)).sum(1))
        return self.classifier(torch.cat(branches, dim=-1))
