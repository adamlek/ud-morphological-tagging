from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
import torch
import os
import pickle
import random
from IPython import embed
from itertools import chain
from typing import List
from collections import Counter
import numpy as np

device = torch.device('cuda:0')

np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)

class XLMRUDDataloader():
    """Documentation for XLMRDataloader
    """
    def __init__(self, train_iter, dev_iter, test_iter, tokens, lemmas, features):
        super(XLMRUDDataloader, self).__init__()
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.tokens = tokens
        self.lemmas = lemmas
        self.features = features
        self.chars = self.char_vocab(tokens)

    def iters(self):
        return self.get_iter(self.train_iter), self.get_iter(self.dev_iter), self.get_iter(self.test_iter)

    def char_vocab(self, token_field):
        char_voc = {'<pad>':0, '<unk>':1}
        for word in token_field.vocab.itos:
            if word in ['<pad>', '<unk>']:
                pass
            else:
                for ch in set(word):
                    if ch not in char_voc:
                        char_voc[ch] = len(char_voc)
        return char_voc
    
    def get_iter(self, _iter):    
        for batch in _iter:
            c, cl = batch.tokens
            q, ql = batch.lemmas
            targets = batch.features

            sentences = self.model_encode(c)
            out = {'sentences': sentences,
                   'tokens': c,
                   'tokens_len': cl,
                   'targets':targets,
                   'char_features':self.get_char_idxs(c)}

            yield out
            
    def get_char_idxs(self, text):
        batch_char_idxs = []
        bcx = list(map(lambda x: list(self.tokens.vocab.itos[x]), chain.from_iterable(text)))
        bcx = list(map(lambda x: '<pad>' if x == ['<', 'p', 'a', 'd', '>'] else x, bcx))

        max_lens = [len(x) for x in bcx]
        batch_words = torch.zeros(len(bcx), max(max_lens)).long()

        for i, word in enumerate(bcx):
            if word != '<pad>':
                batch_words[i, :max_lens[i]] = torch.Tensor([self.chars[x] for x in word])
                
        return batch_words

    def model_encode(self, text):
        #seq = torch.zeros([text.size(0), text.size(1), 1024]).to(device)
        sentences = []
        batch_alignments = []
        for i in range(text.size(0)):
            sentence_string = ' '.join([self.tokens.vocab.itos[x]
                                        for x in text[i] if x != 1])
            sentences.append(sentence_string)
            
        return sentences

def dataloader(ddir, treebank, batch_size):
    whitespacer = lambda x: x.split(' ')
    
    TOKENS = Field(tokenize=whitespacer,
                   lower=True,
                   include_lengths=True,
                   batch_first=True,
                   pad_token='<pad>')
    
    LEMMAS= Field(tokenize=whitespacer,
                   lower=True,
                   include_lengths=True,
                   batch_first=True,
                   pad_token='<pad>')
    
    FEATURES = Field(tokenize=whitespacer,
                     batch_first=True,
                     pad_token='<pad>')
    
    train, dev, test = TabularDataset.splits(path=ddir+treebank,
                                             train='train.csv',
                                             validation='dev.csv',
                                             test='test.csv',
                                             format='csv',
                                             fields=[('tokens', TOKENS),
                                                     ('lemmas', LEMMAS),
                                                     ('features', FEATURES)],
                                             skip_header=True,
                                             csv_reader_params={'delimiter':'\t',
                                                                'quotechar':'½'})

    TOKENS.build_vocab(train, dev, test)
    LEMMAS.build_vocab(train, dev, test)
    FEATURES.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = BucketIterator.splits((train, dev, test),
                                                            batch_size=batch_size,
                                                            sort_within_batch=True,
                                                            sort_key=lambda x: len(x.tokens),
                                                            shuffle=True,
                                                            device=device)

    return train_iter, dev_iter, test_iter, TOKENS, LEMMAS, FEATURES

"""
    def features_to_words(self,
                          features: torch.Tensor,
                          alignment: List[List[int]]) -> torch.Tensor:
        assert features.dim() == 2

        bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
        assert bpe_counts[0] == 0  # <s> shouldn't be aligned
        # !!! !!! !!!
        denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
        weighted_features = features / denom.unsqueeze(-1)

        output = [weighted_features[0]]
        largest_j = -1
        for bpe_indices in alignment:
            output.append(weighted_features[bpe_indices].sum(dim=0))
            largest_j = max(largest_j, *bpe_indices)
        for j in range(largest_j + 1, len(features)):
            output.append(weighted_features[j])
        output = torch.stack(output)

        # random assert errors...
        try:
            assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
        except:
            pass
            
        return output
"""

if __name__ == '__main__':
    """
    train, dev, test, tokens, lemma, upos, xpos = dataloader('UD_English-EWT', 'en_ewt-ud')

    ud_loader = XLMRUDDataloader(train, dev, test, tokens, lemma, upos, xpos)

    train_iter, dev_iter, test_iter = ud_loader.iters()

    for i, x in enumerate(train_iter):
        print(i)
        print(x)
    """
    #for dir in os.listdir('data/'):
    #    pass
