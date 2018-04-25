#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random
import math

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu, SequenceNoise
import opts

from onmt.TrainerPortion import Trainer

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid[0]:
    #cuda.set_device(opt.gpuid[0])
    opt.gpuid = [torch.cuda.current_device()]
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(opt.tensorboard_log_dir, comment="Onmt")

print(vars(opts))

def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
                iprint('checkpoint')
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            report_stats.log_tensorboard("progress", writer, lr, epoch)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, data_hook=None, portion_shards=None, start_portion=1):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.data_hook = data_hook
        self.portion_shards = portion_shards
        self.pcnt = start_portion - 1
        print(portion_shards)
        print(self.pcnt)
        self.scnt = 1

        self.cur_iter = '' #self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        self.cur_iter = self._next_dataset_iterator(dataset_iter)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch

            if self.is_train and self.portion_shards[self.pcnt] <= self.scnt:
                print(self.portion_shards)
                print(self.pcnt)
                self.pcnt += 1
                print("Starting portion {}".format(self.pcnt+1))
                self.scnt = 1
                yield None
            self.cur_iter = self._next_dataset_iterator(dataset_iter)
            self.scnt += 1
            

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields
       
        if self.data_hook is not None:
            self.cur_dataset.examples = self.data_hook( self.cur_dataset.examples )

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True, data_hook=None, 
                        portion_shards=None, start_portion=1):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            return sofar + max(len(new.tgt), len(new.src)) + 1

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, data_hook, portion_shards, start_portion)


def make_loss_compute(model, tgt_vocab, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing)

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, fields, optim, data_type, model_opt):
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    #trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
    trainer = Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count, 
                           detach_encoder=model_opt.detach_encoder)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * number of portions: %d, starting from portion %d ' %
          (opt.train_portions, opt.start_portion))
    print(' * batch size: %d' % opt.batch_size)
    print('pswap {}; pdrop {}; pinsert {}; reverse src {}'.format(opt.pswap, opt.pdrop, opt.pinsert, opt.reverse_src))
    
    if opt.pswap + opt.pdrop + opt.pinsert + opt.reverse_src + opt.reverse_tgt > 0:
        noiser = SequenceNoise(opt.pswap, opt.pdrop, opt.pinsert, fields["src"].vocab, opt.reverse_src, opt.reverse_tgt)
        data_hook = noiser.noise_examples 
    else:
        data_hook = None

    shards = len(sorted(glob.glob(opt.data + '.train.[0-9]*.pt')))
    size = math.ceil(shards / opt.train_portions)
    tp = opt.train_portions
    short = size*tp - shards
    portion_shards = [size if i >= short else size-1 for i in range(tp)]
    portion_shards.reverse()
    assert sum(portion_shards) == shards

    start_portion = opt.start_portion
   
    shard_cnt = 0
    if start_portion > 1:
        shard_cnt += opt.start_portion
        print(shard_cnt)
        portion_idx = 0
        while opt.start_portion > sum( portion_shards[:portion_idx] ):
            portion_idx += 1
        print("Portion idx {}".format(portion_idx))
    else:
        portion_idx = 1

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('New Epoch {}'.format(epoch))

        # 1. Train for one epoch on the training set.
        print("Start portion: {}".format(start_portion))
        train_iter = make_dataset_iter(lazily_load_dataset("train", start_portion),
                                       fields, opt, data_hook=data_hook,
                                       portion_shards=portion_shards,
                                       start_portion=portion_idx)
        
        for i, scnt in enumerate(portion_shards[portion_idx-1:]):
            
            train_stats = trainer.train(train_iter, epoch, report_func, scnt)
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())

            # 2. Validate on the validation set.
            if opt.reverse_src + opt.reverse_tgt > 0:
                valid_noiser = SequenceNoise(0, 0, 0, fields["src"].vocab, opt.reverse_src, opt.reverse_tgt)
                valid_data_hook = valid_noiser.noise_examples 
            else:
                valid_data_hook = None
            
            valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                           fields, opt,
                                           is_train=False,
                                           data_hook=valid_data_hook)
            valid_stats = trainer.validate(valid_iter)
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())
       
            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)
        
            # 5. Drop a checkpoint if needed.
            shard_cnt += scnt
            if epoch >= opt.start_checkpoint_at:
                trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, train_portion=shard_cnt)
        
        start_portion = 1
        portion_idx = 1
        shard_cnt = 0
        
        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type, start_portion=1):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    pts = pts[start_portion-1:]
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim


def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.

        shards = len(sorted(glob.glob(opt.data + '.train.[0-9]*.pt')))
        if 'train_portion' in checkpoint:
            opt.start_portion = ( checkpoint['train_portion'] % shards ) + 1
        else:
            opt.start_portion = 1
            
        if opt.start_portion == 1:
            opt.start_epoch = checkpoint['epoch'] + 1
        else:
            opt.start_epoch = checkpoint['epoch']
        print(opt.start_epoch, opt.start_portion)
    
    else:
        checkpoint = None
        model_opt = opt
    
    if not hasattr(model_opt, 'share_rnn'):
        model_opt.share_rnn = False
    if not hasattr(model_opt, 'detach_encoder'):
        model_opt.detach_encoder = False
    
    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = load_fields(first_dataset, data_type, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_model(model, fields, optim, data_type, model_opt)

    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
