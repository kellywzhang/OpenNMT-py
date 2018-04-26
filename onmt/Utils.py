import torch
import numpy as np


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


class SequenceNoise(object):
    def __init__(self, pswap, pdrop, pinsert, vocab, \
                 reverse_src=False, reverse_tgt=False, reverse_order=False):
        self.pswap = pswap
        self.pdrop = pdrop
        self.pinsert = pinsert
        self.vocab = vocab
        self.reverse_src = reverse_src
        self.reverse_tgt = reverse_tgt
        self.reverse_order = reverse_order
        self.max_vocab = len(vocab) - 1

    def noise_examples(self, examples):
        for ex in examples:
            seq = list(ex.src)
            if self.pdrop > 0:
                drop_prob = np.random.binomial(1, self.pdrop, len(seq))
                if sum(drop_prob) < len(seq):
                    seq = [x for i, x in enumerate(seq) if drop_prob[i] != 1]

            if self.pinsert > 0:
                insert_prob = np.random.binomial(1, self.pinsert, len(seq))
                rand_words = [self.vocab.itos[ np.random.randint(2, self.max_vocab) ] for i in range( sum(insert_prob) )]
                rcnt = 0
                for i in range( len(seq)-1, -1, -1):
                    if insert_prob[i] == 1:
                        seq.insert( i, rand_words[rcnt] )
                        rcnt += 1

            if self.pswap > 0:
                swap_prob = np.random.binomial(1, self.pdrop, len(seq)//2)
                even = [x for i, x in enumerate(seq) if i % 2 == 0]
                odd = [x for i, x in enumerate(seq) if i % 2 == 1]
                pairs = list(zip(even, odd))
                nseq = sum( [[y, x] if swap_prob[i] == 1 else [x, y] for i, (x, y) in enumerate(pairs)], [] )
                if len(seq) % 2 == 1:
                    nseq.append( seq[-1] )
                seq = nseq

            if self.reverse_src:
                seq.reverse()

            ex.src = seq

            if self.reverse_tgt:
                tseq = list(ex.tgt)
                tseq.reverse()
                ex.tgt = tseq

            if self.reverse_order:
                src = ex.src
                tgt = ex.tgt
                ex.tgt = src
                ex.src = tgt

        return examples
