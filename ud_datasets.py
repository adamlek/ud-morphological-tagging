from IPython import embed
import os
import pprint
import torch
from collections import defaultdict
import numpy as np

#pretrained_model = torch.hub.load('pytorch/fairseq', 'xlmr.large')
#pretrained_model.to(device)
#pretrained_model.eval()

def read_conll(in_path, out_path):
    out = open(out_path, '+w')

    with open(in_path) as f:
        tokens, sentences, bpe_tokens = 0, 0, 0
        blens = []
        sentence = {'words':[],
                    'lemmas':[],
                    'features':[]}
        
        for line in f.readlines():
            if line.startswith('#'):
                continue
            elif line == '\n':
                if len(sentence['words']) > 300:
                    #print(sentences)
                    pass
                else:
                    entry = '\t'.join([' '.join(sentence['words']).lower(),
                                       ' '.join(sentence['lemmas']).lower(),
                                       ' '.join(sentence['features'])+'\n'])
                    sentences += 1
                    out.write(entry)
                    #bpe = pretrained_model.encode(' '.join(sentence['words'])[1:-1])
                    #bpe_tokens += len(bpe)
                    #blens.append(len(bpe))
                    
                    
                sentence = {'words':[],
                            'lemmas':[],
                            'features':[]}
            else:
                tokens += 1
                _, word, lemma, _, _, features, *_ = line.split('\t')
                sentence['words'].append(word_fix(word))
                sentence['lemmas'].append(lemma)
                features = ';'.join(sorted(set(features.split(';'))))
                sentence['features'].append(features)
                
    out.close()
    #print(np.mean(blens))
    return tokens, sentences, bpe_tokens

def word_fix(word):
    if word == '….':
        return '...'
    elif word == '…':
        return '...'
    elif ' ' in word:
        return word.replace(' ','')
    elif word == '½':
        return '1/2'
    elif 'º' in word:
        return word.replace('º', 'o')
    elif word == 'ª':
        return 'a'
    elif 'ª' in word:
        return word.replace('ª', 'a')
    elif '´' in word:
        return word.replace('´', "'")
    elif word.endswith(' '):
        return word[:-1]
    elif word == '№':
        return 'no.'
    elif word == 'm²':
        return 'm2'
    elif word == 'km²':
        return 'km2'
    elif word == '²':
        return '2'
    elif word == '¾':
        return '3/4'
    elif word == '﻿"':
        return '"'
    elif word == '³':
        return '3'
    else:
        return word

def new_datasets():
        #ud_path = '/home/adamek/data/enhanced-dependencies/train-dev/'
    ud_path = '/home/adamek/git/ud-morphological-tagging/2019/task2/'
    out_path = 'data/'

    tb_bpe = defaultdict(float)

    f_langs = ['UD_Basque-BDT']#, 'UD_Czech-CAC', 'UD_German-GSD', 'UD_Urdu-UDTB', 'UD_Polish-LFG', 'UD_Latvian-LVTB', 'UD_Swedish-LinES']
    #stats = open('features-stats.csv', '+w')
    #stats.write('\t'.join(['treebank','dataset','tokens','sentences'])+'\n')
    for lang in sorted(filter(lambda x: x.startswith('UD'), os.listdir(ud_path))):
        lt, lb = 0, 0
        #if lang not in f_langs:
        #    continue
        #print('reading', lang)
        
        for d in os.listdir(ud_path+lang):
            if not os.path.isdir(out_path+'/'+lang):
                os.mkdir(out_path+'/'+lang)
                
            if d.endswith('.conllu') and 'covered' not in d:
                full_out_path = out_path+'/'+lang+f'/{d.split("-")[-1].split(".")[0]}.csv'
                t, s, b = read_conll(ud_path+lang+'/'+d,
                                     full_out_path)
                print(full_out_path, t, s, b, b/t)
                lt += t
                lb += b
                #return
                #stats.write('\t'.join([lang, d.split("-")[-1].split(".")[0], str(t), str(s)])+'\n')
        tb_bpe[lang] = lb/lt
        #print(lb/lt)
    #stats.close()

    #for k, v in sorted(tb_bpe.items(), key=lambda x: x[1], reverse=True):
    #    print(k, v)

def explore():
    path = '/home/adamek/git/ud-morphological-tagging/data/UD_Basque-BDT/dev.csv'
    all_tags = set()
    with open(path) as f:
        for line in f.readlines():
            words, lemmas, tags = line.split('\t')
            words = words.split(' ')
            lemmas = lemmas.split(' ')
            tags = tags.split(' ')

            for tag in tags:
                all_tags.add(tag)

    print(treebank, len(all_tags))



if __name__ == '__main__':
    #morph_stuff('/home/adamek/data/UniversalDependencies-2.5/UD/UD_Finnish-TDT/fi_tdt-ud-train.conllu')
    new_datasets()
    #explore()
