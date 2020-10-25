import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from IPython import embed
from torchcrf import CRF
import math
from typing import List
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from itertools import chain

device = torch.device('cuda:0')
from args import args

class ContextAttention(nn.Module):
    """Documentation for MSDAttention
    - positional embeddings?
    """
    def __init__(self, dim):
        super(ContextAttention, self).__init__()
        self.t1 = nn.Linear(dim, dim, bias=False)
        self.t2 = nn.Linear(dim, dim, bias=False)
        self.t3 = nn.Linear(dim, dim, bias=False)
        
    def forward(self, token: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(
            torch.matmul(
                self.t1(token), self.t2(token).transpose(-2,-1))/math.sqrt(token.size(-1)),
            dim=-1)
        out = torch.matmul(scores, self.t3(token))
        return out

class CharLSTM(nn.Module):
    """Documentation for CharLSTM
    """
    def __init__(self, num_chars):
        super(CharLSTM, self).__init__()
        self.char_embeddings = nn.Embedding(num_chars, args.char_dim)
        self.char_lstm = nn.GRU(args.char_dim,
                                args.char_dim,
                                bidirectional=args.char_bidir,
                                batch_first=True)

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        embedded_chars = self.char_embeddings(words)
        _, hx = self.char_lstm(embedded_chars)
        return torch.cat([hx[0,:,:], hx[1,:,:]], -1)

class MorphologicalTagger(nn.Module):
    """Documentation for MorphologicalTagger
    """
    def __init__(self, tags, num_tokens, num_chars, mode):
        super(MorphologicalTagger, self).__init__()
        self.tags = tags
        self.mode = mode

        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base', output_hidden_states=True)
        
        self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlmr_model = XLMRobertaModel.from_pretrained('xlm-roberta-base', config=config)
        
        if args.xlmr_size == 'xlmr.large':
            self.lm_dim = 1024
        else:
            self.lm_dim = 768

        if args.char_bidir:
            self.c_dim = args.char_dim*2
        else:
            self.c_dim = args.char_dim

        if args.use_char_representations:
            self.in_dim = self.lm_dim + self.c_dim
            self.out_dim = self.lm_dim + self.c_dim
        else:
            self.in_dim = self.lm_dim
            self.out_dim = self.lm_dim

        if args.use_char_representations:
            self.char_rnn = CharLSTM(num_chars)
            self.char_dropout = nn.Dropout(args.char_dropout)

        if args.use_word_lstm:
            self.word_lstm = nn.LSTM(self.in_dim,
                                     self.out_dim,
                                     bidirectional=args.word_bidir,
                                     batch_first=True)

        if args.use_word_lstm and args.word_bidir:
            self.in_dim *= 2
            self.out_dim *= 2

        self.a = nn.ReLU()
        
        self.bpe2word = BPE2Word(self.lm_dim, mode)
        self.bpe_dropout = nn.Dropout(args.bpe_dropout)
        
        self.transform_dropout = nn.Dropout(args.transform_dropout)
        self.msd_transform = nn.Sequential(self.transform_dropout,
                                           #nn.Linear(self.out_dim, self.out_dim, bias=True),
                                           #nn.LeakyReLU(),
                                           nn.Linear(self.out_dim, self.out_dim, bias=True),
                                           nn.GELU())

        self.msd_tagger = nn.Sequential(nn.Linear(self.out_dim, self.out_dim, bias=True),
                                        nn.GELU(),
                                        nn.Linear(self.out_dim, tags))
        #self.context_attention = ContextAttention(self.out_dim)
        
        
    def forward(self, sentences, tokens, char_features):#, alignment):
        # add prediction mask

    #with torch.autograd.profiler.profile() as prof:
        if args.use_char_representations:
            char_repr = self.char_rnn(char_features)
            char_repr = self.char_dropout(char_repr)

        bpe_features, alignment = self.xlmr_prepare_features(sentences)
        bpe_features, _, layer_features = self.xlmr_model(bpe_features)
        layer_features = torch.stack(list(layer_features), 1)
        
        sequence, pred_mask = self.bpe2word(layer_features, alignment)
        sequence = self.bpe_dropout(sequence)
        #transformed_sequence = self.word_dropout(transformed_sequence)

        if args.use_char_representations:
            sequence = torch.cat([sequence,
                                  char_repr.view(sequence.size(0),
                                                 sequence.size(1), self.c_dim)],
                                 -1)
            sequence = self.a(sequence)

        if args.use_word_lstm:
            sequence, _ = self.word_lstm(sequence)

        #embed()
        #assert False

        #sequence = self.context_attention(sequence)

        # residual conntection
        representation = sequence + self.msd_transform(sequence)
        #representation = self.msd_transform(sequence)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            
        # log_softmax over predictions (for NLLLoss)
        return F.log_softmax(self.msd_tagger(representation), -1), alignment

    def word_dropout(self, seqs, p=0.3):
        bs, sl, dim = seqs.size(0), seqs.size(1), seqs.size(2)
        seqs = seqs.flatten().view(bs*sl,dim)*(torch.randn(bs*sl,1).uniform_().to(device)>p).float()
        return seqs.view(bs, sl, dim)

    def xlmr_train(self, train=True):
        if train:
            self.xlmr_model.train()
            #pass
        else:
            pass
            self.xlmr_model.eval()
    
    def xlmr_prepare_features(self, sentences):
        bpes = []
        alignments = []
        for sentence in sentences:
            bpe_tokens, alignment = self.word_tokenization(sentence)
            bpes.append(bpe_tokens)
            alignments.append(alignment)

        sentence_sizes = max([x.size(1) for x in bpes])
        bpe_batch = torch.ones(len(bpes), sentence_sizes).long().to(device)
        for i, instance in enumerate(bpes):
            # add word-dropout
            bpe_batch[i,:instance.size(1)] = instance[0,:].to(device)

        return bpe_batch.to(device), alignments
        
    def word_tokenization(self,
                      sentence: str,
                      return_all_hiddens: bool = True) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""


        bpe_strs = list(map(lambda x: x[1].replace('▁',''), enumerate(self.xlmr_tokenizer.tokenize(sentence))))
        bpe_toks = torch.tensor(self.xlmr_tokenizer.encode(sentence)).unsqueeze(0).to(device)
        tokens = sentence.split(' ')
        #embed()

        alignment = self.align_bpe_to_words(bpe_strs,
                                            tokens)

        return bpe_toks, alignment


    def input_dropout(self, seq, p=args.xlmr_dropout):
        x = (torch.randn(len(seq)).uniform_()>p).float().to(device)
        x[0] = 1.
        x[-1] = 1.
        bpe_toks = torch.where(x.float()==1.,
                               seq,
                               torch.Tensor([self.xlmr_tokenizer.unk_token_id]).long().to(device))
        return bpe_toks
        
    def align_bpe_to_words(self,
                           bpe_tokens: List[str],
                           other_tokens: List[str]):
        # create alignment from every word to a list of BPE tokens
        alignment = []
        bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=0))
        j, bpe_tok = next(bpe_toks)
        for other_tok in other_tokens:
            bpe_indices = []
            while True:
                if other_tok.startswith(bpe_tok):
                    bpe_indices.append(j)
                    other_tok = other_tok[len(bpe_tok):]
                    try:
                        j, bpe_tok = next(bpe_toks)
                    except StopIteration:
                        j, bpe_tok = None, None
                elif bpe_tok.startswith(other_tok):
                    # other_tok spans multiple BPE tokens
                    bpe_indices.append(j)
                    bpe_tok = bpe_tok[len(other_tok):]
                    other_tok = ''
                else:
                    # fix <unk>-token
                    raise Exception(
                        'Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                if other_tok == '':
                    break
            assert len(bpe_indices) > 0
            alignment.append(bpe_indices)
        assert len(alignment) == len(other_tokens)


        for j, al in enumerate(alignment):
            bi = ''.join([bpe_tokens[x] for x in al])
            assert bi == other_tokens[j]
            #print(bi)
            
        return alignment

class BPE2Word(nn.Module):
    """Documentation for BPE2Word
    """
    def __init__(self, xlmr_dim, mode):
        super(BPE2Word, self).__init__()
        self.mode = mode
        self.xlmr_dim = xlmr_dim

        if args.xlmr_size == 'xlmr.large':
            self.layers = 25
        else:
            self.layers = 13

        # initialize layer weights uniformly
        #self.layer_w = nn.Parameter(torch.Tensor([1.]*self.layers), requires_grad=True)
        self.layer_w = nn.Parameter(torch.randn(self.layers), requires_grad=True)
        self.layer_c = nn.Parameter(torch.randn(1), requires_grad=True)
        self.layer_dropout = torch.ones(self.layers).to(device)
        self.layer_dropout_p = args.layer_dropout
        self.layer_repr_dropout = nn.Dropout(args.layer_repr_dropout)
        
        if mode in [4, 5]:
            self.bpe_self_attention = BPEAttention(self.xlmr_dim, mode)
        elif mode == 3:
            #self.lstm = nn.LSTM(1024, 1024, bidirectional=True, batch_first=True)
            self.ln1 = nn.LayerNorm(self.xlmr_dim, elementwise_affine=False)
            self.rnn = nn.LSTM(self.xlmr_dim, self.xlmr_dim, bidirectional=True, batch_first=True)
        elif mode in [7,8,9]:
            self.feature_transform = nn.Sequential(nn.Linear(self.xlmr_dim, self.xlmr_dim),
                                                   nn.ReLU())

    def layer_attention(self, bpe_features: List[torch.Tensor]) -> List[torch.Tensor]:
        w = torch.mul(self.layer_w, self.layer_dropout)
        layer_att = torch.mul(F.softmax(w, -1), bpe_features.permute(0,2,3,1)).permute(0,3,1,2).sum(1)
        return layer_att
        
    def forward(self, bpe_features: List[torch.Tensor], alignment: List[List[int]]) -> torch.Tensor:
        
        max_len = max([len(x) for x in alignment])
        #max_len=1
        # layer dropout, set all representations of layer j to 0.
        self.layer_dropout = torch.mul(self.layer_dropout,
                                       (torch.randn(self.layers).uniform_().to(device)>self.layer_dropout_p).float()).to(device)

        # apply layer attention, weighted sum of all layers
        bpe_features = self.layer_attention(bpe_features)
        
        if self.mode == 1:
            return self.bpe_sum(bpe_features, alignment, max_len)
        elif self.mode == 2:
            return self.bpe_mean(bpe_features, alignment, max_len)
        elif self.mode == 3:
            return self.bpe_lstm(bpe_features, alignment, max_len)
        elif self.mode in [4, 5]:
            return self.bpe_attention(bpe_features, alignment, max_len)
        elif self.mode == 6:
            return self.bpe_max(bpe_features, alignment, max_len)
        elif self.mode == 7:
            return self.bpe_first(bpe_features, alignment, max_len)
        elif self.mode == 8:
            return self.bpe_sum_withparameters(bpe_features, alignment, max_len)
        elif self.mode == 9:
            return self.bpe_mean_withparameters(bpe_features, alignment, max_len)
        else:
            pass

    # bpe_features[i]
    def bpe_sum(self,
                bpe_features: List[torch.Tensor],
                alignment: List[List[int]],
                max_len: int) -> torch.Tensor:

        batch_features = torch.zeros(len(alignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            #bpe_t = self.transform(bpe_features[i].to(device))
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_features[i,token[0],:]
                else:
                    batch_features[i,j,:] = bpe_features[i,token[0]:token[-1]+1,:].sum(0)

        return batch_features, None

    def bpe_sum_withparameters(self,
                               bpe_features: List[torch.Tensor],
                               alignment: List[List[int]],
                               max_len: int) -> torch.Tensor:

        batch_features = torch.zeros(len(walignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            #bpe_t = self.transform(bpe_features[i].to(device))
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_features[i,token[0],:]
                else:
                    features_transformed = self.feature_transform(
                        bpe_features[i,token[0]:token[-1]+1,:])
                    batch_features[i,j,:] = features_transformed.sum(0)
                    #batch_features[i,j,:] = bpe_features[i,token[0]:token[-1]+1,:].sum(0)

        return batch_features, None
        
    def bpe_mean(self,
                 bpe_features: List[torch.Tensor],
                 alignment: List[List[int]],
                 max_len: int) -> torch.Tensor:

        batch_features = torch.zeros(len(alignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            #bpe_t = self.transform(bpe_features[i].to(device))
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_features[i,token[0],:]
                else:
                    batch_features[i,j,:] = bpe_features[i,token[0]:token[-1]+1,:].mean(0)
    
        return batch_features, None

    def bpe_mean_withparameters(self,
                                bpe_features: List[torch.Tensor],
                                alignment: List[List[int]],
                                max_len: int) -> torch.Tensor:

        batch_features = torch.zeros(len(alignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_features[i,token[0],:]
                else:
                    features_transformed = self.feature_transform(
                        bpe_features[i,token[0]:token[-1]+1,:])
                    batch_features[i,j,:] = features_transformed.mean(0)

        return batch_features, None

    def bpe_lstm(self,
                bpe_features: List[torch.Tensor],
                alignment: List[List[int]],
                max_len: int) -> torch.Tensor:

        num_words = [len(x) for x in chain.from_iterable(alignment)]
        batch_features = torch.zeros(len(num_words), max(num_words), self.xlmr_dim).to(device)

        #batch_features = torch.zeros(len(bpe_features), max_len, self.xlmr_dim).to(device)
        #alignment_lens = [[len(z) for z in x] for x in alignment]
        #ml = max([max([len(z) for z in x]) for x in alignment])

        wc = 0
        for i, sentence in enumerate(alignment):
            for word in sentence:
                try:
                    if len(word) == 1:
                        batch_features[wc,0,:] = bpe_features[i, word[0], :]
                    else:
                        batch_features[wc,:len(word),:] = bpe_features[i,word[0]:word[-1]+1,:]
                except:
                    print('bpe2rnn_input error')
                    embed()
                    assert False
                wc += 1

        # reconstruct batch
        #rnn_input = self.bpe_dropout(batch_features)
        _, (hx, cx) = self.rnn(batch_features)
        sentence_lens = [len(a) for a in alignment]
        output_features = torch.zeros(bpe_features.size(0), max(sentence_lens), self.xlmr_dim*2).to(device)
        curr_i = 0
        for i, x in enumerate(sentence_lens):
            try:
                #word_repr = hx[:,curr_i:curr_i+x,:].sum(0)
                word_repr = torch.cat([hx[0,curr_i:curr_i+x,:], hx[1,curr_i:curr_i+x,:]], 1)
                #embed()
                #assert False
                #word_repr = hx[:,curr_i:curr_i+x,:]
                output_features[i,:abs(curr_i - (curr_i+x)),:] = word_repr
                curr_i += x
            except:
                print('batch reconstruction error')
                embed()
                assert False
        
        return output_features, None
    
    def bpe_attention(self,
                      bpe_features: List[torch.Tensor],
                      alignment: List[List[int]],
                      max_len: int) -> torch.Tensor:
        
        batch_features = torch.zeros(len(bpe_features), max_len, self.xlmr_dim).to(device)
        for i, sentence in enumerate(alignment):
            #bpe_t = self.transform(bpe_features[i].to(device))
            bpe_t = bpe_features[i].to(device)
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_t[token[0]]
                else:
                    batch_features[i,j,:] = self.bpe_self_attention(
                        bpe_t[token[0]:token[-1]+1,:].unsqueeze(1).to(device))
        return batch_features, None


    def bpe_max(self,
                 bpe_features: List[torch.Tensor],
                 alignment: List[List[int]],
                 max_len: int) -> torch.Tensor:

        batch_features = torch.zeros(len(alignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_features[i,token[0],:]
                else:
                    batch_features[i,j,:] = torch.max(bpe_features[i,token[0]:token[-1]+1,:], 0)[0]
    
        return batch_features, None
    
    def bpe_max_old(self,
                bpe_features: List[torch.Tensor],
                alignment: List[List[int]],
                max_len: int) -> torch.Tensor:
        
        batch_features = torch.zeros(len(bpe_features), max_len, self.xlmr_dim).to(device)
        for i, sentence in enumerate(alignment):
            #bpe_t = self.transform(bpe_features[i].to(device))
            bpe_t = bpe_features[i].to(device)
            for j, token in enumerate(sentence):
                if len(token) == 1:
                    batch_features[i,j,:] = bpe_t[token[0]]
                else:
                    batch_features[i,j,:] = torch.max(bpe_t[token[0]:token[-1]+1,:].unsqueeze(0),
                                                      1)[0].to(device)
        return batch_features, None

    def bpe_first(self,
                  bpe_features: List[torch.Tensor],
                  alignment: List[List[int]],
                  max_len: int) -> torch.Tensor:
        
        batch_features = torch.zeros(len(alignment), max_len, self.xlmr_dim).to(device)

        for i, sentence in enumerate(alignment):
            for j, token in enumerate(sentence):
                first_token = self.feature_transform(bpe_features[i,token[0],:])
                batch_features[i,j,:] = first_token
    
        return batch_features, None
    
class BPEAttention(nn.Module):
    """Documentation for BPEAttention
    - positional embeddings?
    """
    def __init__(self, xlmr_dim, mode):
        super(BPEAttention, self).__init__()
        self.mode = mode
        self.xlmr_dim = xlmr_dim
        self.lin1 = nn.Linear(self.xlmr_dim, self.xlmr_dim, bias=False)
        
        if mode == 5:
            self.lin2 = nn.Linear(self.xlmr_dim, self.xlmr_dim, bias=False)
            self.lin3 = nn.Linear(self.xlmr_dim, self.xlmr_dim, bias=False)
            
    def forward(self, token: torch.Tensor) -> torch.Tensor:
        if self.mode == 5:
            scores = F.softmax(
                torch.matmul(
                    self.lin1(token), self.lin2(token).transpose(-2,-1))/math.sqrt(self.xlmr_dim),
                dim=-1)
            out = torch.matmul(
                scores, self.lin3(token)).permute(1,0,2).sum(1)
        else:
            scores = F.softmax(
                torch.matmul(
                    self.lin1(token), token.transpose(-2,-1))/math.sqrt(self.xlmr_dim),
                dim=-1)
            out = torch.matmul(
                scores, token).permute(1,0,2).sum(1)
        #embed()
        #assert False
        return out

"""
    def get_prediction_mask(self,
                            alignment: List[List[int]],
                            max_len: int) -> torch.Tensor:
        
        #max_len = max([sum([len(z) for z in x]) for x in alignment])
        mask = torch.zeros(len(alignment), max_len)
        
        for i, sentence in enumerate(alignment):
            mask_s = []
            for token in sentence:
                if len(token) == 1:
                    mask_s.append(1)
                else:
                    t = [0 for x in token]
                    t[-1] = 1
                    mask_s += t
            mask[i,:len(mask_s)] = torch.Tensor(mask_s)
        return mask.int()

    def bpe_rnn(self,
                bpe_features: List[torch.Tensor],
                alignment: List[List[int]],
                max_len: int) -> torch.Tensor:

        ml = max([x.size(0) for x in bpe_features])
        prediction_mask = self.get_prediction_mask(alignment, ml)
        
        batch_features = torch.zeros(len(bpe_features), ml, 512).to(device)
        
        for i in range(len(bpe_features)):
            batch_features[i,:bpe_features[i].size(0),:] = bpe_features[i].unsqueeze(0).to(device)

        return batch_features, prediction_mask

    """
    
