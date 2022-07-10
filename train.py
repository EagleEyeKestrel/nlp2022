import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.legacy import data
from model import POSTagger
from utils import epoch_time, read_json
from pytorch_pretrained_bert import BertAdam, BertTokenizer

SEED = 1024
BATCH_SIZE = 16
EPOCHS = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

WORDS = data.Field()
TAGS = data.Field()

fields = {"words": ("words", WORDS), "tags": ("tags", TAGS)}
path = ""
#path = "/content/drive/MyDrive/nlp2022/"
train_data, valid_data = data.TabularDataset.splits(
    path=path+"data",
    train="train.json",
    validation="valid.json",
    format="json",
    fields=fields,
)

print(len(train_data))
# vec = torchtext.vocab.Vectors(name=path+".vector_cache/sgns.wiki.word")
# WORDS.build_vocab(train_data, vectors=vec, unk_init=torch.Tensor.normal_)
TAGS.build_vocab(train_data)

# print(len(WORDS.vocab))
# print(TAGS.vocab.stoi)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_ids_train, input_masks_train, labels_train = read_json(path + "data/train.json",
                                                             path + "bert/chinese_roberta_wwm_ext_pytorch/",
                                                             TAGS.vocab.stoi)
input_ids_valid, input_masks_valid, labels_valid = read_json(path + "data/valid.json",
                                                             path + "bert/chinese_roberta_wwm_ext_pytorch/",
                                                             TAGS.vocab.stoi)
train_data = torch.utils.data.TensorDataset(torch.LongTensor(input_ids_train),
                                            torch.LongTensor(input_masks_train),
                                            torch.LongTensor(labels_train))
train_sampler = torch.utils.data.sampler.RandomSampler(train_data)
train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
valid_data = torch.utils.data.TensorDataset(torch.LongTensor(input_ids_valid),
                                            torch.LongTensor(input_masks_valid),
                                            torch.LongTensor(labels_valid))
valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_data)
valid_loader = torch.utils.data.DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]
model = POSTagger(path + "bert/chinese_roberta_wwm_ext_pytorch/", len(TAGS.vocab))


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_loader) * EPOCHS
                    )
model = model.to(device)


def compute_acc(pred, std):
    pred = pred.argmax(dim=1)
    #print(pred.shape, std.shape)
    correct = pred.eq(std)
    return correct.sum() / (torch.FloatTensor([pred.shape[0]]).to(device))


def train(model, loader):
    epoch_loss, epoch_acc = 0, 0
    model.train()
    for batch_idx, (words, mask, tags) in enumerate(loader):
        optimizer.zero_grad()
        words, mask, tags = words.to(device), mask.to(device), tags.to(device)
        pred = model([words, mask])
        #pred: [batch size, len, output dim]
        #tags: [batch size, len]
        pred = pred.view(-1, pred.shape[-1])
        tags = tags.view(-1)
        loss = F.cross_entropy(pred, tags)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc = compute_acc(pred, tags)
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader):
    epoch_loss, epoch_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (words, mask, tags) in enumerate(loader):
            optimizer.zero_grad()
            words, mask, tags = words.to(device), mask.to(device), tags.to(device)
            pred = model([words, mask, tags])
            pred = pred.view(-1, pred.shape[-1])
            tags = tags.view(-1)
            loss = F.cross_entropy(pred, tags)
            epoch_loss += loss.item()
            acc = compute_acc(pred, tags)
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)




best_loss = float('inf')
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_loader)
    valid_loss, valid_acc = evaluate(model, valid_loader)
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), path + 'postag-model.pt')
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

tokenizer = BertTokenizer(vocab_file=path + "bert/chinese_roberta_wwm_ext_pytorch/vocab.txt")
model.load_state_dict(torch.load(path + "postag-model.pt"))

def predict_test(model, sentence):
    model.eval()
    words, word_ids = [], []
    for ch in sentence:
        words.append(ch)
    for t in words:
        word_ids.append(100 if t not in tokenizer.vocab else tokenizer.vocab[t])
    t1 = torch.LongTensor(word_ids).to(device)
    t1 = t1.unsqueeze(0)
    mask = [1 for _ in range(len(words))]
    t2 = torch.LongTensor(mask).to(device)
    t2 = t2.unsqueeze(0)
    pred = model([t1, t2])
    pred = pred.view(-1, pred.shape[-1])
    pred = pred.argmax(dim=1)
    tags = [TAGS.vocab.itos[tag] for tag in pred]
    txt = ''
    assert len(words) == len(tags)
    for i in range(len(words)):
        txt += words[i]
        tag = tags[i]
        if tag.find('-') != -1:
            pos = tag.index('-')
            tag1 = tag[0:pos]
            if tag[-1] == 'E' or tag[-1] == 'S':
                txt += tag1
                if i != len(words) - 1:
                    txt += ' '
    return txt



test_file = open(path + "EvaHan_testa_raw.txt", 'r', encoding='utf-8')
output_file = open(path + "EvaHan_testa_result.txt", 'w', encoding='utf-8')
for sentence in test_file.readlines():
    if len(sentence) <= 1:
        continue
    sentence = sentence[0:-1]
    txt = predict_test(model, sentence)
    output_file.write(txt)
    output_file.write('\n')
test_file.close()
output_file.close()

test_file = open(path + "EvaHan_testb_raw.txt", 'r', encoding='utf-8')
output_file = open(path + "EvaHan_testb_result.txt", 'w', encoding='utf-8')
for sentence in test_file.readlines():
    if len(sentence) <= 1:
        continue
    sentence = sentence[0:-1]
    txt = predict_test(model, sentence)
    output_file.write(txt)
    output_file.write('\n')
test_file.close()
output_file.close()