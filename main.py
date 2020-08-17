import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import pprint
from dataloader import XLMRUDDataloader, dataloader
from IPython import embed
from sklearn.metrics import f1_score
from model import MorphologicalTagger
from collections import Counter, defaultdict
from args import args
import pytorch_warmup as warmup
import sklearn.metrics as m

device = torch.device('cuda:0')


np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)

def get_opts(model, num_steps):
    opt = optim.AdamW([
        {'params': model.bpe2word.parameters()}, #xlmr-lr here?
        {'params': model.msd_tagger.parameters()},
        {'params': model.msd_transform.parameters()},
        {'params': model.word_lstm.parameters()},
        {'params': model.char_rnn.parameters()},
        #{'params': model.context_attention.parameters()},
        {'params': model.xlmr_model.parameters(), 'lr': args.xlmr_lr}
    ], lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    
    #lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=args.min_lr)
    return opt, lr_scheduler
    
def xlmr_main(treebank, mode):
    ddir = 'data/'
    train_i, dev_i, test_i, tokens, lemma, features = dataloader(ddir,
                                                                 treebank,
                                                                 args.batch_size)

    ud_loader = XLMRUDDataloader(train_i, dev_i, test_i, tokens, lemma, features)

    num_labels = len(features.vocab)
    num_tokens = len(tokens.vocab)
    model = MorphologicalTagger(num_labels, num_tokens, len(ud_loader.chars), mode).to(device)
    num_params = sum([p.numel() for p in model.parameters()])

    criterion = nn.NLLLoss()#ignore_index=0)
    #criterion2 = nn.NLLLoss(reduction='mean')
    criterion2 = nn.KLDivLoss(reduction='batchmean')

    if args.lr_schedule_restarts:
        num_steps = len(train_i)# * args.n_epochs
    else:
        num_steps = len(train_i) * args.n_epochs
        
    opt, lr_scheduler = get_opts(model, num_steps)
    #warmup_scheduler = warmup.LinearWarmup(opt, warmup_period=len(train_i))
    #warmup_scheduler.last_step = -1
    warmup_scheduler = 0
    #print(model)
    dev_accs = []
    patience_counter = 0

    print(treebank)
    print(f'Number of MSD tags: {num_labels}, Number of characters: {len(ud_loader.chars)}')
    print('params:', num_params)
    print(args)
    best_dev_score = 0
    
    #init_params(model, nn.init.xavier_normal_, gain=nn.init.calculate_gain('relu'))

    for epoch in range(args.n_epochs):
        train_iter, dev_iter, test_iter = ud_loader.iters()
        
        train_loss, train_acc, train_sacc = train(train_iter,
                                                  model,
                                                  criterion,
                                                  criterion2,
                                                  opt,
                                                  lr_scheduler,
                                                  warmup_scheduler,
                                                  epoch,
                                                  num_labels)
        if args.use_lr_scheduler:
            if args.lr_schedule_restarts:
                # lr-schedule with restart after every epoch
                opt, lr_scheduler = get_opts(model, num_steps)
        
        dev_loss, dev_acc, dev_sacc = validate(dev_iter, model, criterion, criterion2)
        dev_accs.append(dev_acc)

        # save model if acc >= previous best model
        if dev_acc >= best_dev_score:
            torch.save(model.state_dict(), f'./models/{treebank.lower()}-{mode}.pt')
            best_dev_score = dev_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(epoch, ':',
              np.round(train_loss,3), np.round(dev_loss, 3), ':', 
              np.round(train_acc, 3), np.round(dev_acc, 3), ':',
              c) # Accuracy for complete sentences

        # early stopping
        if args.early_stopping:
            if patience_counter == args.early_stopping_patience:
                break
        
    model.load_state_dict(torch.load(f'./models/{treebank.lower()}-{mode}.pt'))
    test_loss, test_acc, test_sacc, outp = validate(test_iter,
                                                    model,
                                                    criterion,
                                                    criterion2,
                                                    test=True)

    bpe_accuracy = bpe_len_accuracy(outp)
    
    print(f'TOKEN ACCURACY: {np.round(test_acc,3)} SENT ACCURACY: {np.round(test_sacc, 3)}')
    
    return np.round(test_acc, 3), 0#np.round(large_bpes['correct']/large_bpes['total'], 3)

def bpe_len_accuracy(outp):
    bpe_accuracies = defaultdict(list)
    for pred, gold, alignment in outp:
        for y_hat, y, bpe_tokens in zip(pred, gold, alignment):
            bpe_accuracies[len(bpe)].append(y_hat==y)
    return bpe_accuracies
    

def init_params(model, init_func, *params, **kwargs):
    for name, p in model.named_parameters():
        if 'weight' in name and 'ln' not in name:
            init_func(p, *params, **kwargs)

def label_smoothing(labels, num_labels, a=args.label_smoothing):
    to_one_hot = torch.eye(num_labels).to(device)
    labels = to_one_hot[labels]
    return (1 - a) * labels + a / num_labels
            
def train(data_iter, model, criterion, criterion2, opt, lr_scheduler, warmup_scheduler, epoch, num_labels):
    model.train()
    model.xlmr_train(True)
    token_correct, token_total, e_loss = 0, 0, 0
    sent_correct, sent_total = 0, 0

    # dont train xlm-r the first epoch
    if epoch == 0:
        for n, p in model.named_parameters():
            if n.startswith('xlmr'):
                p.requires_grad = False
        #model.xlmr_model.requires_grad = False
    else:
        for n, p in model.named_parameters():
            if n.startswith('xlmr'):
                p.requires_grad = True
        #model.xlmr_model.requires_grad = True
    
    for i, batch in enumerate(data_iter):
        #lr_scheduler.step(epoch-1)
        #warmup_scheduler.dampen()
            
        sentences = batch['sentences']
        tokens = batch['tokens'].to(device)
        features = batch['targets'].to(device)
        char_features = batch['char_features'].to(device)
        
        output, alignment = model(sentences, tokens, char_features)#, alignment)

        targets = label_smoothing(features, num_labels)
        
#        loss_v2 = criterion2(output.contiguous().view(-1, output.size(-1)),
#                             features.contiguous().view(-1))
        loss_v2 = criterion2(output.contiguous().view(-1, output.size(-1)),
                             targets.contiguous().view(-1, targets.size(-1)))
        #embed()
        #assert False

        e_loss += loss_v2.item()

        #pred_nopad, gold_nopad = remove_pad(torch.argmax(output, -1),
        #                                    features,
        #                                    seq_mask)

        pred = torch.argmax(output.contiguous().view(-1, output.size(-1)),-1) 
        gold = features.view(-1)
        token_correct += (pred==gold).sum().item()
        token_total += pred.view(-1).size().numel() # pred.size(0)

        print(i, np.round(e_loss/(i+1),5),
              np.round(token_correct/token_total, 5),
              end='\r')
        
        loss_v2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping) # 2.5
        opt.step()
        if args.use_lr_scheduler:
            lr_scheduler.step()
        opt.zero_grad()

    return e_loss/(i+1), token_correct/token_total, 0

def validate(data_iter, model, criterion, criterion2, test=False, output_report=False):
    model.eval()
    model.xlmr_train(False)
    correct, total, e_loss = 0, 0, 0
    sent_correct, sent_total = 0, 0
    outp = []
    yp, yt = [], []
    
    for i, batch in enumerate(data_iter):
        sentences = batch['sentences']
        tokens = batch['tokens'].to(device)
        features = batch['targets'].to(device)
        char_features = batch['char_features'].to(device)
        
        with torch.no_grad():
            output, alignment = model(sentences, tokens, char_features)#, alignment)
            #output, alignment = model(bpe_features, tokens, char_features)#, alignment)
            
        #features = label_smoothing(features)
        #embed()
        #assert False    
        loss_v2 = criterion(output.contiguous().view(-1, output.size(-1)),
                            features.contiguous().view(-1))

        loss_v = loss_v2
        e_loss += loss_v.item()

        #pred_nopad, gold_nopad = remove_pad(torch.argmax(output, -1), features, seq_mask)
        pred = torch.argmax(output.contiguous().view(-1, output.size(-1)),-1) 
        gold = features.view(-1)
        correct += (pred==gold).sum().item()
        total += pred.view(-1).size().numel() # pred.size(0)

        yt += gold.tolist()
        yp += pred.tolist()
        
        if test:
            outp.append((pred, gold, list(chain.from_iterable(alignment))))
    
    if test:
        return e_loss/(i+1), correct/total, m.f1_score(yt, yp, average='micro'), outp
        
    return e_loss/(i+1), correct/total, m.f1_score(yt, yp, average='micro')

def remove_pad(pred, gold, lens):
    p, g = torch.empty([lens.flatten().int().sum()]), torch.empty([lens.flatten().int().sum()])
    pos = 0
    for i in range(pred.size(0)):
        pad_start = lens[i,:].sum().int().item()#-2
        p[pos:pos+pad_start] = pred[i,:pad_start]
        g[pos:pos+pad_start] = gold[i,:pad_start]
        pos += pad_start
    return p.int(), g.int()

def sent_acc(pred, gold, seq_mask):
    c = 0
    for i in range(seq_mask.size(0)):
        p, g = pred[i,:].tolist(), gold[i,:].tolist()
        if p == g:
            c += 1
    return c

def dataset_test(treebank):
    ddir = 'data/'
    print(treebank)
    train_i, dev_i, test_i, tokens, lemma, features = dataloader(ddir,
                                                                 treebank,
                                                                 args.batch_size)
    num_steps = len(train_i) * args.n_epochs 

    ud_loader = XLMRUDDataloader(train_i, dev_i, test_i, tokens, lemma, features)

    train_iter, dev_iter, test_iter = ud_loader.iters()

    for i, x in enumerate(train_iter):
        print(i)

if __name__ == '__main__':
    mode_map = {'sum':1,
                'mean':2,
                'rnn':3,
                'att1':4,
                'att3':5,
                'max':6,
                'final':7}
    
    tb_scores = {}
    
    for treebank in ['UD_Basque-BDT',
                    'UD_Finnish-TDT',
                    'UD_Arabic-PADT',
                    'UD_Spanish-AnCora',
                    'UD_Turkish-IMST',
                    'UD_Czech-CAC']:
                    #'UD_Turkish-PUD',
                    #'UD_English-PUD',
                    #'UD_Russian-PUD',
                    #'UD_Swedish-PUD',
                    #'UD_Turkish-PUD',
                    #'UD_Czech-PUD',
                    #'UD_Arabic-PUD',
                    #'UD_Finnish-PUD',
                    #'UD_Estonian-EDT',
                    #'UD_German-GSD',
                    #'UD_Finnish-FTB',]:

        scores = {'token':[], 'bpe':[]}
        for mode in [1,3]:
            print('=======================')
            print('mode =', mode)
            print('=======================')
            token, large_bpe = xlmr_main(treebank, mode)
            print('=======================')
            scores['token'].append(token)
            scores['bpe'].append(large_bpe)

        tb_scores[treebank[3:]] = ' & '.join(
            map(lambda x: str(x), scores['token']+scores['bpe']))+'\\\\'
        
    #with open('latest_run.txt', '+w') as f:
    for k, v in tb_scores.items():
        print(k, ' &   & ' + v)
    
