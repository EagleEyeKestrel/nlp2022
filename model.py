import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class POSTagger(nn.Module):
    def __init__(self, bert_path, output_dim):
        super().__init__()
        self.bert_path = bert_path
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, output_dim)

    def forward(self, item):
        words, mask = item[0], item[1]
        #print(words.shape, mask.shape)
        feats, pooled = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        #print(feats.shape, pooled.shape)
        res = self.fc(feats)
        #print(res.shape)
        return res
