import torch
import torch.nn as nn
import torch.nn.functional as F


class POSTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding_layer = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        #[len, batch size]
        embedded = self.dropout(self.embedding_layer(words))
        #[len, batch size, embedding dim]
        output, (h, c) = self.lstm(embedded)
        #[len, batch size, hid * 2]
        res = self.fc(output)
        #[len, batch size, output dim]
        return res
