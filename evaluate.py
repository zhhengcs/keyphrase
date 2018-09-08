from __future__ import print_function
import math
import logging
import string

import nltk
import scipy
import torch
from nltk.stem.porter import *
import numpy as np
from collections import Counter,defaultdict

import os

from torch.autograd import Variable

import config
import pykp
from utils import Progbar
from pykp.metric.bleu import bleu

stemmer = PorterStemmer()


def process_predseqs(pred_seqs, oov, id2word, opt):
    '''
    :param pred_seqs:
    :param src_str:
    :param oov:
    :param id2word:
    :param opt:
    :return:
    '''
    processed_seqs = []
    if_valid = []

    for seq in pred_seqs:

        # convert to words and remove the EOS token

        seq_sentence_np = [int(x.cpu().data.numpy()) for x in seq.sentence]
        processed_seq = [id2word[x] if x < opt.vocab_size else oov[x - opt.vocab_size] for x in seq_sentence_np[:-1]]
        
        keep_flag = True

        if len(processed_seq) == 0:
            keep_flag = False

        if keep_flag and any([w == pykp.io.UNK_WORD for w in processed_seq]):
            keep_flag = False

        if keep_flag and any([w == '.' or w == ',' for w in processed_seq]):
            keep_flag = False

        if_valid.append(keep_flag)
        processed_seqs.append((seq, processed_seq, seq.score))

    unzipped = list(zip(*(processed_seqs)))
    processed_seqs, processed_str_seqs, processed_scores = unzipped if len(processed_seqs) > 0 and len(unzipped) == 3 else ([], [], [])

    assert len(processed_seqs) == len(processed_str_seqs) == len(processed_scores) == len(if_valid)
    return if_valid, processed_seqs, processed_str_seqs, processed_scores


def post_process_predseqs(seqs, num_oneword_seq=1):
    processed_seqs = []

    # -1 means no filter applied
    if num_oneword_seq == -1:
        return seqs

    for seq, str_seq, score in zip(*seqs):
        

        if len(str_seq) == 1 and num_oneword_seq <= 0:
            continue

        processed_seqs.append((seq, str_seq, score))
        # update the number of one-word sequeces to keep
        if len(str_seq) == 1:
            num_oneword_seq -= 1

    unzipped = list(zip(*(processed_seqs)))
    if len(unzipped) != 3:
        return ([], [], [])
    else:
        
        return unzipped


def if_present_duplicate_phrase(src_str, phrase_seqs):
    stemmed_src_str = stem_word_list(src_str)
    present_index = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for phrase_seq in phrase_seqs:
        stemmed_pred_seq = stem_word_list(phrase_seq)

        # check if it is duplicate
        if '_'.join(stemmed_pred_seq) in phrase_set:
            present_index.append(False)
            continue

        # check if it appears in source text
        for src_start_idx in range(len(stemmed_src_str) - len(stemmed_pred_seq) + 1):
            match = True
            for seq_idx, seq_w in enumerate(stemmed_pred_seq):
                src_w = stemmed_src_str[src_start_idx + seq_idx]
                if src_w != seq_w:
                    match = False
                    break
            if match:
                break

        # if it reaches the end of source and no match, means it doesn't appear in the source, thus discard
        if match:
            present_index.append(True)
        else:
            present_index.append(False)
        phrase_set.add('_'.join(stemmed_pred_seq))
    # print(*phrase_set)
    return present_index


def match_pred_and_true(src_str,true_seqs,pred_seqs,do_stem = True,sample_idx =None):
    if do_stem:
        pred_seqs = pred_seqs[0]
        
        true_seqs = stem_word_list(true_seqs)
        pred_seqs = stem_word_list(pred_seqs)

    if len(set(pred_seqs)) != len(pred_seqs):
            print(pred_seqs)
    match_score = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    
    fw = open('result/predict_results.'+str(sample_idx),'w')

    src_str = ' '.join(src_str)
    fw.write('Article\t')
    fw.write(src_str)
    fw.write('\n')

    fw.write('Target phrase:\t')
    t_s = ' '.join(true_seqs)
    fw.write(t_s)
    fw.write('\n')

    fw.write('Pred phrase:\t')
    p_s = ' '.join(pred_seqs)
    fw.write(p_s)
    fw.write('\n')
    fw.write('\n')
    fw.close()

    for idx,w in enumerate(pred_seqs):
        match = False
        for ww in true_seqs:
            if w == ww:
                match = True
                break
        if match:
            match_score[idx] = 1.0

    p_5 = sum(match_score[:5])/5.0
    r_5 = sum(match_score[:5])*1.0/len(true_seqs)
    

    if p_5 + r_5 == 0:
        f_5 = 0
    else:
        f_5 = 2*p_5*r_5/(r_5+p_5)

    
    p_10 = sum(match_score[:10])/10.0
    r_10 = sum(match_score[:10])*1.0/len(true_seqs)
    
    if p_10 + r_10 == 0:
        f_10 = 0
    else:
        f_10 = 2 * p_10*r_10/(p_10+r_10)

    return p_5,r_5,f_5,p_10,r_10,f_10
def logging_result(src,trg_seq_list,pred_trg_seq,match_list):
    print('Article:---------------------------- 0 ------------------------')
    print(' '.join(src))

    print('keyphrase:')
    for trg in trg_seq_list:
        print('\t',trg)
    print('Pred seq:')
    
    for idx,trg in enumerate(pred_trg_seq[:10]):
        if match_list[idx]:
            print('\t',trg,'Correct Keyphrase!')
        else:
            print('\t',trg)
    print('')

def evaluate_beam_search(generator, data_loader, opt, title='', epoch=1, save_path=None):
    logging = config.init_logging(title, save_path + '/%s.log' % title)
    progbar = Progbar(logger=logging, title=title, target=len(data_loader)/opt.beam_batch, batch_size=opt.beam_batch,
                      total_examples=len(data_loader)/opt.beam_batch)

    beam_batch_idx = 0
    score_dict = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}

    sample_idx = 0
    for i, batch in enumerate(data_loader):
        beam_batch_idx += 1

        src_list, src_len, trg_list, _, _, src_oov_map_list, oov_list, query_lists,query_len,src_str_list, trg_str_list = batch
        
        if torch.cuda.is_available() and opt.use_gpu:
            src_list = src_list.cuda()
            src_oov_map_list = src_oov_map_list.cuda()
            query_lists = query_lists.cuda()
        
        pred_seq_list = generator.beam_search(src_list, src_len, src_oov_map_list, oov_list, opt.word2id,query_lists,query_len)        
        
       
        for src, src_str, trg, trg_str_seqs, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, pred_seq_list, oov_list):
            # logging.info('======================  %d =========================' % (beam_batch_idx))
            
            pred_is_valid, processed_pred_seqs, processed_pred_str_seqs, processed_pred_score = process_predseqs(pred_seq, oov, opt.id2word, opt)
            
            # 2nd filtering: if filter out phrases that don't appear in text, and keep unique ones after stemming
            
            pred_is_present = [True] * len(processed_pred_str_seqs)
            
            valid_and_present = np.asarray(pred_is_valid) * np.asarray(pred_is_present)
            
            '''
            Evaluate predictions w.r.t different filterings and metrics
            '''
            num_oneword_seq = -1# -1,1
            topk_range = [5,10] #5,10
            score_names = ['precision', 'recall', 'f_score']
            
            processed_pred_seqs = np.asarray(processed_pred_seqs)[valid_and_present]
            processed_pred_str_seqs = np.asarray(processed_pred_str_seqs)[valid_and_present]
            processed_pred_score = np.asarray(processed_pred_score)[valid_and_present]
            
            # 3rd round filtering (one-word phrases)
            filtered_pred_seq, filtered_pred_str_seqs, filtered_pred_score = post_process_predseqs((processed_pred_seqs, processed_pred_str_seqs, processed_pred_score), num_oneword_seq)
            
            match_list = get_match_result(true_seqs=trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='exact')
            # logging_result(src_str,trg_str_seqs,filtered_pred_str_seqs,match_list)

            assert len(filtered_pred_seq) == len(filtered_pred_str_seqs) == len(filtered_pred_score) == len(match_list)

            for topk in topk_range:
                results = evaluate(match_list, filtered_pred_seq, trg_str_seqs, topk=topk)
                for k, v in zip(score_names, results):
                    if '%s@%d#oneword=%d' % (k, topk, num_oneword_seq) not in score_dict:
                        score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)] = []
                    score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)].append(v)

    
        if beam_batch_idx%10 == 0:
            
            # print('#(precision@5#oneword=1)=%d, avg=%f' % (len(score_dict['precision@5#oneword=1']), np.average(score_dict['precision@5#oneword=1'])))
            # print('#(precision@10#oneword=1)=%d, avg=%f' % (len(score_dict['precision@10#oneword=1']), np.average(score_dict['precision@10#oneword=1'])))
        
            # print('#(recall@5#oneword=1)=%d, avg=%f' % (len(score_dict['recall@5#oneword=1']), np.average(score_dict['recall@5#oneword=1'])))
            # print('#(recall@10#oneword=1)=%d, avg=%f' % (len(score_dict['recall@10#oneword=1']), np.average(score_dict['recall@10#oneword=1'])))
            metric5 = 'f_score@5#oneword='+str(num_oneword_seq)
            metric10 = 'f_score@10#oneword='+str(num_oneword_seq)

            x,y = np.average(score_dict[metric5]),np.average(score_dict[metric10])
            print(metric5,x)
            print(metric10,y) 
            # print('#(f_score@5#oneword=1)=%d, avg=%f' % (len(score_dict['f_score@5#oneword=1']), x))
            # print('#(f_score@10#oneword=1)=%d, avg=%f' % (len(score_dict['f_score@10#oneword=1']), y))

            progbar.update(epoch, beam_batch_idx, [(metric5, x),(metric10, y)])

            print('*'*50)
        
        '''
        process each example in current batch
        '''        
        
    print(metric5,x)
    print(metric10,y)

    
    # if save_path:
    #     # export scores. Each row is scores (precision, recall and f-score) of different way of filtering predictions (how many one-word predictions to keep)
    #     with open(save_path + os.path.sep + title + '_result.csv', 'w') as result_csv:
    #         csv_lines = []
            
    #         for topk in topk_range:
    #             csv_line = '#oneword=%d,@%d' % (num_oneword_seq, topk)
    #             for k in score_names:
    #                 csv_line += ',%f' % np.average(score_dict['%s@%d#oneword=%d' % (k, topk, num_oneword_seq)])
    #             csv_lines.append(csv_line + '\n')

    #         result_csv.writelines(csv_lines)

    return score_dict


def stem_word_list(word_list):
    
    return [stemmer.stem(w.strip().lower()) for w in word_list]

def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 2)
    return precision, recall, f_score

def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact'):
    '''
    :param true_seqs:
    :param pred_seqs:
    :param do_stem:
    :param topn:
    :param type: 'exact' or 'partial'
    :return:
    '''
    micro_metrics = []
    micro_matches = []

    # do processing to baseline predictions
    match_score = np.asarray([0.0] * len(pred_seqs), dtype='float32')
    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:

        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]
    for pred_id, pred_seq in enumerate(pred_seqs):

        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'partial':
            max_similarity = 0.
            pred_seq_set = set(pred_seq)
            # use the jaccard coefficient as the degree of partial match
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                jaccard = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                if jaccard > max_similarity:
                    max_similarity = jaccard
            match_score[pred_id] = max_similarity

        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            match_score[pred_id] = bleu(pred_seq, true_seqs, [0.1, 0.3, 0.6])

    return match_score


def evaluate(match_list, predicted_list, true_list, topk=5):
    if len(match_list) > topk:
        match_list = match_list[:topk]
    if len(predicted_list) > topk:
        predicted_list = predicted_list[:topk]

    # Micro-Averaged  Method
    micropk = float(sum(match_list)) / float(len(predicted_list)) if len(predicted_list) > 0 else 0.0
    micrork = float(sum(match_list)) / float(len(true_list)) if len(true_list) > 0 else 0.0

    if micropk + micrork > 0:
        microf1 = float(2 * (micropk * micrork)) / (micropk + micrork)
    else:
        microf1 = 0.0

    return micropk, micrork, microf1


def f1_score(prediction, ground_truth):
    # both prediction and grount_truth should be list of words
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

