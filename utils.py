import json
from pytorch_pretrained_bert import BertTokenizer
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def read_json(filename, bert_path, tag_dict):
    f = open(filename, 'r', encoding='utf-8')
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")
    input_ids = []
    input_masks = []
    labels = []
    pad_size = 100
    for line in f.readlines():
        mp = json.loads(line)
        words = mp["words"]
        tags = mp["tags"]
        tag_ids = []
        word_ids = []
        for t in words:
            word_ids.append(100 if t not in tokenizer.vocab else tokenizer.vocab[t])
        for t in tags:
            tag_ids.append(0 if t not in tag_dict else tag_dict[t])
        masks = [1] * len(word_ids)
        if len(word_ids) < pad_size:
            word_ids = word_ids + [0] * (pad_size - len(word_ids))
            tag_ids = tag_ids + [0] * (pad_size - len(tag_ids))
            masks = masks + [0] * (pad_size - len(masks))
        else:
            word_ids = word_ids[:pad_size]
            tag_ids = tag_ids[:pad_size]
            masks = masks[:pad_size]
        input_ids.append(word_ids)
        input_masks.append(masks)
        labels.append(tag_ids)
        assert len(word_ids) == len(tag_ids) == len(masks) == pad_size
    return input_ids, input_masks, labels


