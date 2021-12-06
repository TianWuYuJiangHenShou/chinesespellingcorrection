import os,json
import tokenization
import random

def load_pinyin(file):
    with open(file,'r',encoding='utf-8') as f:
        pinyin = f.readlines()

    py_conf = {}
    for line in pinyin:
        key,value = line.split('\t')
        py_conf[key] = value.split()
    return py_conf

def load_stroke(file,vocab_file):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    confusion_datas = {}
    for line in open(file, 'r', encoding='utf-8'):
        line = line.strip().split(',')
        stroke = set()
        for k in line:
            if k in tokenizer.vocab:
                stroke.add(k)
        for k in stroke:
            confusion_datas[k] = list(stroke)
    return confusion_datas,tokenizer

same_pinyin = load_pinyin('../pre_train_src/confusions/same_pinyin.txt')
simi_pinyin = load_pinyin('../pre_train_src/confusions/simi_pinyin.txt')

stroke,tokenizer = load_stroke('../pre_train_src/confusions/same_stroke.txt','../pre_train_src/datas/vocab.txt')
all_vocab = []
for k,v in same_pinyin.items():
    all_vocab += v
all_vocab = list(set(all_vocab))


with open('./datas/pool/test.json','r') as f:
    data = json.load(f)

texts = []
for _,value in data.items():
    text = value['zh']
    texts.append(text)

res = []
for text in texts:
    raw ,masked_sample = [],[]
    for token in list(text):
        if token in tokenizer.vocab:
            raw.append(token)
            percent = random.random()
            if percent < 0.9:
                masked_sample.append(token)
                continue
            else:
                prob = random.random()
                if prob <= 0.3:
                    #pinyin
                    value = same_pinyin.get(token,None)
                    if value is None or len(value) < 2:
                        masked_sample.append(token)
                    else:
                        index = random.randint(0,len(value) - 1)
                        masked_sample.append(value[index])
                elif prob <= 0.6:
                    # same_pinyin
                    value = simi_pinyin.get(token,None)
                    if value is None or len(value) < 2:
                        masked_sample.append(token)
                    else:
                        index = random.randint(0,len(value) - 1)
                        masked_sample.append(value[index])
                elif prob <= 0.75:
                    # stroke
                    value = stroke.get(token,None)
                    if value is None or len(value) < 2:
                        masked_sample.append(token)
                    else:
                        index = random.randint(0,len(value) - 1)
                        masked_sample.append(value[index])
                elif prob <= 0.85:
                    # random
                    masked_sample.append(all_vocab[random.randint(0,len(all_vocab) - 1)])
                else:
                    masked_sample.append(token)
    #print(''.join(raw) + '\n')
    #print(''.join(masked_sample) + '\n')
    #print('\n')
    res.append((' '.join(raw),' '.join(masked_sample)))

f = open('./datas/finance_test.txt','w')
for source,mask in res:
    f.writelines(mask + '\t' + source + '\n')




