f = open("/Users/jiluyang/Desktop/zuozhuan.json", 'r', encoding='utf-8')
train = open("/Users/jiluyang/Desktop/train.json", 'w', encoding='utf-8')
valid = open("/Users/jiluyang/Desktop/valid.json", 'w', encoding='utf-8')
lines = f.readlines()
train_size = len(lines) * 8 // 10
valid_size = len(lines) - train_size
for i in range(train_size):
    train.write(lines[i])
for i in range(valid_size):
    valid.write(lines[i + train_size])
train.close()
valid.close()