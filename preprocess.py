#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch

import config
import pykp.io
import json

parser = argparse.ArgumentParser(
    description='preprocess.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")


parser.add_argument('-dataset', default='kp20k',
                    help="Name of dataset")
parser.add_argument('-save_data', default='AAAI',
                    help="Output file for the prepared data")


parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-seed', type=int, default=9527,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

config.preprocess_opts(parser)
opt = parser.parse_args()
# opt.train_path = 'data/%s/%s_training.json'     % (opt.dataset, opt.dataset)
opt.train_path = 'data/AAAI/small_train.json'
opt.test_path = 'data/AAAI/small_test.json'
# opt.valid_path = 'data/%s/%s_validation.json'   % (opt.dataset, opt.dataset)
# opt.test_path  = 'data/%s/%s_testing.json'      % (opt.dataset, opt.dataset)
# opt.save_data  = 'data/%s/%s' % (opt.save_data, opt.dataset)
opt.save_data = 'data/AAAI/'

def main():
    '''
    Load and process training data
    '''
    # load keyphrase data from file, each data example is a pair of (src_str, [kp_1, kp_2 ... kp_m])

    if opt.dataset == 'kp20k':
        src_fields = ['title', 'abstract']
        trg_fields = ['keyword']
    elif opt.dataset == 'stackexchange':
        src_fields = ['title', 'question']
        trg_fields = ['tags']
    else:
        raise Exception('Unsupported dataset name=%s' % opt.dataset)

    print("Loading training data...")
    src_trgs_pairs = pykp.io.load_json_data(opt.train_path, name=opt.dataset, src_fields=src_fields, trg_fields=trg_fields, trg_delimiter=';')
   
    print("Processing training data...")
    tokenized_train_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize = pykp.io.copyseq_tokenize, opt=opt, valid_check=True)


    def json_save(sample_list,file_name):
        fw = open(file_name,'w')
        for pair in sample_list:
            stt = json.dumps(pair)
            fw.write(stt)
            fw.write('\n')
        fw.close()
    
    
    print("Building Vocab...")
    word2id, id2word, vocab = pykp.io.build_vocab(tokenized_train_pairs, opt)
   
    torch.save([word2id, id2word, vocab],open(opt.save_data + 'vocab.pt', 'wb'))
    print('Vocab size = %d' % len(vocab))
    
    # word2id,id2word,vocab = torch.load(open(opt.save_data + 'vocab.pt'))

    # print("Building training...")
    train_one2one = pykp.io.build_dataset(tokenized_train_pairs, word2id, id2word, opt, mode='one2one',save_path=opt.save_data+'train.one2one.json')
    # train_one2many = pykp.io.build_dataset(tokenized_train_pairs,word2id,id2word,opt,mode='one2many',include_original=True)
    # print('#pairs of train_one2one = %d' % len(train_one2one))
    # print("Dumping train one2one to disk: %s" % (opt.save_data + '.train.one2one.pt'))
    json_save(train_one2one,)
    # json_save(train_one2many,opt.save_data+'train.one2many.json')

    

    '''
    Load and process test data
    '''
    print("Loading test data...")
    src_trgs_pairs = pykp.io.load_json_data(opt.test_path, name=opt.dataset, src_fields=src_fields, trg_fields=trg_fields, trg_delimiter=';')

    print("Processing test data...")
    tokenized_test_pairs = pykp.io.tokenize_filter_data(
        src_trgs_pairs,
        tokenize=pykp.io.copyseq_tokenize, opt=opt, valid_check=True)
    
    print("Building testing...")
    
    test_one2many = pykp.io.build_dataset(
        tokenized_test_pairs, word2id, id2word, opt, mode='one2many', include_original=True)
    
    json_save(test_one2many, opt.save_data + 'test.one2many.json')
    print("Dumping done!")
    return
 
    '''
    dump to disk
    '''
    
   
    
if __name__ == "__main__":
    main()
