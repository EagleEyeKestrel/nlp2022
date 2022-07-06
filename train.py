import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy import data
from model import POSTagger
from utils import epoch_time

SEED = 1024
BATCH_SIZE = 64
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

WORDS = data.Field()
TAGS = data.Field()

fields = {"words": ("words", WORDS), "tags": ("tags", TAGS)}

train_data, valid_data = data.TabularDataset.splits(
    path="/content/drive/MyDrive/nlp2022/data",
    train="train.json",
    validation="valid.json",
    format="json",
    fields=fields,
)

print(len(train_data))
vec = torchtext.vocab.Vectors(name="/content/drive/MyDrive/nlp2022/.vector_cache/sgns.wiki.word")
WORDS.build_vocab(train_data, vectors=vec, unk_init=torch.Tensor.normal_)
TAGS.build_vocab(train_data)

# print(len(WORDS.vocab))
# print(TAGS.vocab.stoi)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.words),
    sort_within_batch=True,
    device=device
)
UNK_IDX = WORDS.vocab.stoi[WORDS.unk_token]
PAD_IDX = WORDS.vocab.stoi[WORDS.pad_token]
TAG_PAD_IDX = TAGS.vocab.stoi[TAGS.pad_token]
model = POSTagger(len(WORDS.vocab), 300, 128, len(TAGS.vocab),
                  2, True, 0.5, WORDS.vocab.stoi[PAD_IDX])


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


model.apply(init_weights)
model.embedding_layer.weight.data.copy_(WORDS.vocab.vectors)
model.embedding_layer.weight.data[UNK_IDX] = torch.zeros(300)
model.embedding_layer.weight.data[PAD_IDX] = torch.zeros(300)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX)
model = model.to(device)
criterion = criterion.to(device)


def compute_acc(pred, std, TAG_PAD_IDX):
    pred = pred.argmax(dim=1, keepdim=True)
    valid_pos = (std != TAG_PAD_IDX).nonzero()
    correct = pred[valid_pos].squeeze(1).eq(std[valid_pos])
    return correct.sum() / (torch.FloatTensor([std[valid_pos].shape[0]]).to(device))


def train(model, iterator):
    epoch_loss, epoch_acc = 0, 0
    model.train()
    for batch in iterator:
        text, tags = batch.words, batch.tags
        optimizer.zero_grad()
        # [len, batch size]
        pred = model(text)
        # [len, batch size, output dim]
        pred = pred.view(-1, pred.shape[-1])
        tags = tags.view(-1)
        loss = criterion(pred, tags)
        acc = compute_acc(pred, tags, TAG_PAD_IDX)
        loss.backward()
        optimizer.step()
        epoch_acc += acc.item()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):
    epoch_loss, epoch_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, tags = batch.words, batch.tags
            optimizer.zero_grad()
            pred = model(text)
            pred = pred.view(-1, pred.shape[-1])
            tags = tags.view(-1)
            loss = criterion(pred, tags)
            acc = compute_acc(pred, tags, TAG_PAD_IDX)
            epoch_acc += acc.item()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


EPOCHS = 10
best_loss = float('inf')
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator)
    valid_loss, valid_acc = evaluate(model, valid_iterator)
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), '/content/drive/MyDrive/nlp2022/postag-model.pt')
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def predict_test(model, sentence):
    model.eval()
    words = []
    for ch in sentence:
        words.append(ch)
    word_ids = [WORDS.vocab.stoi[word] for word in words]
    tensor = torch.LongTensor(word_ids).to(device)
    tensor = tensor.unsqueeze(1)
    pred = model(tensor)
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


test_file = open("/content/drive/MyDrive/nlp2022/EvaHan_testa_raw.txt", 'r', encoding='utf-8')
output_file = open("/content/drive/MyDrive/nlp2022/EvaHan_testa_result.txt", 'w', encoding='utf-8')
for sentence in test_file.readlines():
    if len(sentence) <= 1:
        continue
    sentence = sentence[0:-1]
    txt = predict_test(model, sentence)
    output_file.write(txt)
    output_file.write('\n')
test_file.close()
output_file.close()

test_file = open("/content/drive/MyDrive/nlp2022/EvaHan_testb_raw.txt", 'r', encoding='utf-8')
output_file = open("/content/drive/MyDrive/nlp2022/EvaHan_testb_result.txt", 'w', encoding='utf-8')
for sentence in test_file.readlines():
    if len(sentence) <= 1:
        continue
    sentence = sentence[0:-1]
    txt = predict_test(model, sentence)
    output_file.write(txt)
    output_file.write('\n')
test_file.close()
output_file.close()