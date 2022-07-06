import json

f = open("/Users/jiluyang/Desktop/zuozhuan.txt", 'r', encoding='utf-8')
output = open('/Users/jiluyang/Desktop/zuozhuan.json', 'w', encoding='utf-8')
for line in f.readlines():
    elements = line.split()
    words, tags, ws, combine = [], [], [], []
    flag = 1
    if len(elements) == 0:
        flag = 0
    for element in elements:
        if element.find('/') == -1:
            flag = 0
            break
    if flag == 0:
        continue
    for element in elements:
        mid = element.index('/')
        token = element[0:mid]
        tag = element[mid:]

        for word in token:
            words.append(word)
            tags.append(tag)
        if len(token) == 1:
            ws.append('S')
        else:
            ws.append('B')
            for _ in range(len(token) - 2):
                ws.append('I')
            ws.append('E')
    text = ''.join(words)
    for i in range(len(words)):
        item = tags[i] + '-' + ws[i]
        combine.append(item)
    d = {'text': text, 'words': words, 'tags': combine}
    encoded_dict = json.dumps(d, ensure_ascii=False, indent=None)
    output.write(encoded_dict)
    output.write('\n')
    assert(len(words) == len(tags))
    assert(len(words) == len(ws))
