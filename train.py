# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os
import sys
import argparse
import commands
import logging
import numpy as np
import time
#import torchtext
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
import evaluate
import utils
import copy

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from evaluate import evaluate_beam_search, get_match_result

from utils import Progbar, plot_learning_curve

import pykp
from pykp.io import KeyphraseDataset
from pykp.model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionCascading
from pykp.dataloader import BucketIterator,KeyphraseDataset
import time
import re


__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def train_ml(one2one_batch, model, optimizer, criterion, opt):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists,query,query_len = one2one_batch
    
    # query_src = src[:,:5]

    #
    max_oov_number = max([len(oov) for oov in oov_lists])

    if torch.cuda.is_available() and opt.use_gpu:
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()
        query = query.cuda()

    optimizer.zero_grad()
  
    decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists,query=query,query_len=query_len)
    
    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()

    if not opt.copy_attention:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size),
            trg_target.contiguous().view(-1)
        )
    else:
        # print(max_oov_number)
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size + max_oov_number),
            trg_copy_target.contiguous().view(-1)
        )
    # loss = loss * (1 - opt.loss_scale)
    # print("--loss calculation- %s seconds ---" % (time.time() - start_time))

    loss.backward()
    # print("--backward- %s seconds ---" % (time.time() - start_time))

    if opt.max_grad_norm > 0:
        pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        
    optimizer.step()

    return loss.data.item(), decoder_log_probs

def train_model(model, optimizer_ml, criterion, train_data_loader,  opt,eval_dataloader):

    logging.info('======================  Start Training  =========================')

    checkpoint_names = []
    train_ml_history_losses = []
    train_rl_history_losses = []
    valid_history_losses = []
    test_history_losses = []
    # best_loss = sys.float_info.max # for normal training/testing loss (likelihood)
    best_loss = 0.0  # for f-score
    stop_increasing = 0

    train_ml_losses = []
    
    total_batch = -1
    early_stop_flag = False
    st = time.time()
    for epoch in range(opt.start_epoch, opt.epochs):
        
        batch_per_epoch = len(train_data_loader)/opt.batch_size
        for batch_i, batch in enumerate(train_data_loader):
            
            model.train()
            total_batch += 1
            one2one_batch = batch

            # Training
            if opt.train_ml:
                loss_ml, decoder_log_probs = train_ml(one2one_batch, model, optimizer_ml, criterion, opt)
                train_ml_losses.append(loss_ml)

            if batch_i % 100 == 0:
                # report(report_loss)
                t = time.time()
                print('Training: Epoch=%d - %d/%d loss:%.4f,cost %fs'%(epoch,batch_i,batch_per_epoch,loss_ml,t-st))
                st = t
            # if total_batch > 1 and total_batch % opt.save_model_every  == 0:  # epoch >= opt.start_checkpoint_at and
            # Save the checkpoint
                
        save_dir = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d' % (opt.exp, epoch, batch_i) + '.model')
        try:
            torch.save(model.state_dict(),open(save_dir, 'wb'))
        except:
            pass

        # evaluate_per_epoch(model,eval_dataloader,opt,epoch)
              

def load_data_vocab(opt, load_train=True):

    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load(opt.vocab + 'vocab.pt', 'wb')
    
    
    # one2one data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)


    logging.info('======================  Dataset  =========================')
    # one2many data loader
    if load_train:
    	
        train_one2one_loader = BucketIterator('./data/AAAI/train.one2one.json',word2id,id2word,
                                            batch_size=opt.batch_size,mode='keyword',
                                            repeat=False,sort=False,
                                            shuffle=False,
                                            Data_type=KeyphraseDataset)
        
        test_one2many_loader = BucketIterator('./data/AAAI/test.one2many.json',word2id,id2word,
                                            batch_size=opt.beam_batch,
                                            include_original=True,
                                            mode='keyphrase',
                                            repeat=False,
                                            sort=False,
                                            shuffle=False,
                                            
                                            Data_type=KeyphraseDataset)

        logging.info('#(train data size:  #(one2one pair)=%d, #(batch)=%d' % (len(train_one2one_loader), len(train_one2one_loader) / train_one2one_loader.batch_size))
    else:	
        train_one2many_loader = None
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    opt.vocab_size = len(id2word)
    
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return train_one2one_loader, word2id, id2word, vocab,test_one2many_loader


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
   
    criterion = torch.nn.NLLLoss(ignore_index=opt.word2id[pykp.io.PAD_WORD])

    if opt.train_ml:
        optimizer_ml = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        optimizer_ml = None


    if torch.cuda.is_available() and opt.use_gpu:
        criterion = criterion.cuda()

    return optimizer_ml, criterion


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.cascading_model:
        model = Seq2SeqLSTMAttentionCascading(opt)
    else:
        model = Seq2SeqLSTMAttention(opt)


    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        
        # load the saved the meta-model and override the current one
        # model = torch.load(
        #     open(os.path.join(opt.model_path, opt.exp, '.initial.model'), 'wb')
        # )

        if torch.cuda.is_available() and opt.use_gpu:
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict([(k[7:], v) if k.startswith('module.') else (k, v) for k, v in checkpoint.items()])
        model.load_state_dict(checkpoint)
    else:
        # dump the meta-model
        meta_model_dir = os.path.join(opt.train_from[: opt.train_from.find('.epoch=')], 'initial.model')
        
        torch.save(
            model.state_dict(),
            open(meta_model_dir, 'wb')
        )

    if torch.cuda.is_available() and opt.use_gpu:

        model = model.cuda()

    utils.tally_parameters(model)
    # embedding = torch.load('embedding40004.pt')
    # model.init_embedding(embedding,requires_grad=False)
    
    return model


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'


    if hasattr(opt, 'copy_attention') and opt.copy_attention:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
        

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)


    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        path = os.path.join(opt.model_path, opt.exp + '.initial.json')
        
        json.dump(vars(opt), open(path, 'w'))

    return opt

def evaluate_per_epoch(model,eval_dataloader,opt,epoch):
    generator = SequenceGenerator(model,opt,eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length,
                                      )


    evaluate_beam_search(generator, eval_dataloader, opt, title='predict',
                     save_path=opt.pred_path + '/epoch=%s' % (epoch))
    
def make_embedding(word2id,id2word):
    f = open('wordvec/glove.6B.100d.txt')
    word2vec = dict()
    for line in f:
        L = line.split()
        w = L[0]
        vec = [float(x) for x in L[1:]]
        word2vec[w] = np.array(vec)
    embedding = torch.zeros((len(word2id),100))
    for i in range(len(word2id)):
        w = id2word[i]
        if w in word2vec:
            
            embedding[i] = torch.from_numpy(word2vec[w])
        else:
            embedding[i].uniform_(-0.1, 0.1)
    torch.save(embedding,'embedding'+str(len(word2id))+'.pt') 
    return embedding

def main():
    # load settings for training
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    config.transformer_opts(parser)

    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    logging = config.init_logging(logger_name=None, log_file=opt.exp_path + '/output.log', stdout=True)
    try:
       
        # opt.train_from = 'model/kp20k.ml.copy.bi-directional.20180901-025437/kp20k.ml.copy.bi-directional.epoch=9.batch=938.model'
        train_data_loader,word2id, id2word, vocab,eval_dataloader = load_data_vocab(opt)
        model = init_model(opt)
        # embedding=make_embedding(word2id,id2word)
        embedding = torch.load('embedding50004.pt')
        model.init_embedding(embedding)

        opt.learning_rate = 0.001
        optimizer_ml,criterion = init_optimizer_criterion(model, opt)
        train_model(model, optimizer_ml, criterion, train_data_loader,  opt,eval_dataloader)

    except Exception as e:
        logging.exception("message")


if __name__ == '__main__':
    main()
