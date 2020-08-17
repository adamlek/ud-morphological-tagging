from argparse import ArgumentParser

a_parser = ArgumentParser()

## model architecture
# representations based on characters
a_parser.add_argument('--use_char_representations', type=bool, default=True)
a_parser.add_argument('--char_dim', type=int, default=128)
a_parser.add_argument('--char_bidir', type=bool, default=True)
# representations based on words
a_parser.add_argument('--use_word_lstm', type=bool, default=True)
a_parser.add_argument('--word_bidir', type=bool, default=True)
# XLM-R model
a_parser.add_argument('--xlmr_size', type=str, default='xlmr.base')

## learning rate
a_parser.add_argument('--lr', type=float, default=0.001)
a_parser.add_argument('--xlmr_lr', type=float, default=0.000001)
a_parser.add_argument('--use_lr_scheduler', type=bool, default=True)
a_parser.add_argument('--lr_schedule_restarts', type=bool, default=True)
a_parser.add_argument('--min_lr', type=float, default=0.000000000001)

## regularization
a_parser.add_argument('--weight_decay', type=float, default=0.05)
# dropout values
a_parser.add_argument('--xlmr_dropout', type=float, default=0.2) # number of input tokens changed to <unk>
a_parser.add_argument('--layer_dropout', type=float, default=0.1) # p that layer j is set to 0.
a_parser.add_argument('--layer_repr_dropout', type=float, default=0.4) # dropout for w (in all layers)
a_parser.add_argument('--transform_dropout', type=float, default=0.5) # dropout before predict(w)
a_parser.add_argument('--char_dropout', type=float, default=0.4) # dropout on w after char rnn
a_parser.add_argument('--bpe_dropout', type=float, default=0.4) # dropout on w after bpe composition
# early stopping
a_parser.add_argument('--early_stopping', type=bool, default=True)
a_parser.add_argument('--early_stopping_patience', type=float, default=2)

## other hyperparams
a_parser.add_argument('--n_epochs', type=int, default=10)
a_parser.add_argument('--batch_size', type=int, default=4)
a_parser.add_argument('--clipping', type=float, default=5.0)
a_parser.add_argument('--label_smoothing', type=float, default=0.03) # 0.0 = no smoothing

## etc
a_parser.add_argument('--seed', type=int, default=333)
a_parser.add_argument('--cuda_device', type=int, default=0)

args = a_parser.parse_args()
