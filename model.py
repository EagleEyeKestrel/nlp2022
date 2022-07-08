import torch
import torch.nn as nn
from torchcrf import CRF


class POSTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.padding_idx = pad_idx
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(output_dim)

    def forward(self, words, tags):
        mask = words != self.padding_idx
        #[len, batch size]
        embedded = self.dropout(self.embedding_layer(words))
        #[len, batch size, embedding dim]
        output, (h, c) = self.lstm(embedded)
        #[len, batch size, hid * 2]
        lstm_feats = self.fc(output)
        #[len, batch size, output dim]
        # print(mask[0].all())
        # if not mask[0].all():
        #     print(mask)
        loss = -self.crf.forward(lstm_feats, tags, mask)
        seq = self.crf.decode(lstm_feats, mask)
        padded_seq = torch.nn.utils.rnn.pad_sequence([torch.tensor(t, dtype=torch.long) for t in seq], batch_first=True)
        #[batch size, len]
        #seq_tensor = torch.tensor(seq, dtype=torch.long)
        seq_tensor = padded_seq.permute(1, 0)
        #[len, batch size]
        return loss, seq_tensor