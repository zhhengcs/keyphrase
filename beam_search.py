"""
Class for generating sequences
Adapted from
https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py
https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/beam_search.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import heapq
# from Queue import PriorityQueue

import sys
import torch
from torch.autograd import Variable

import pykp
from pykp.eric_layers import GetMask
import numpy as np
import collections
import itertools
import logging

from torch.distributions import Categorical


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, batch_id, sentence, dec_hidden, context, ctx_mask, src_oov, oov_list, logprobs, score, attention=None):
        """Initializes the Sequence.

        Args:
          batch_id: Original id of batch
          sentence: List of word ids in the sequence.
          dec_hidden: Model state after generating the previous word.
          logprobs:  The log-probabilitu of each word in the sequence.
          score:    Score of the sequence (log-probability)
        """
        self.batch_id = batch_id
        self.sentence = sentence
        self.vocab = set(sentence)  # for filtering duplicates
        self.dec_hidden = dec_hidden
        self.context = context
        self.ctx_mask = ctx_mask
        self.src_oov = src_oov
        self.oov_list = oov_list
        self.logprobs = logprobs
        self.score = score
        self.attention = attention

    '''
    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1
    '''

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN_heap(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def __len__(self):
        assert self._data is not None
        return len(self._data)

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False,topk=None):
        """Extracts all elements from the TopN.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        if sort:
            data.sort(reverse=True)
        # if topk == 1:
        #     return data[0]
        return data[:topk]

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,opt,
                 eos_id=None,
                 beam_size=3,
                 max_sequence_length=5,
                 return_attention=True,
                 length_normalization_factor=0.0,
                 length_normalization_const=5.,
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_id: the token number symobling the end of sequence
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.model = model
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor
        self.length_normalization_const = length_normalization_const
        self.return_attention = return_attention
        self.get_mask = GetMask()
        self.opt = opt

    def sequence_to_batch(self, sequence_lists):
        '''
        Convert K sequence objects into K batches for RNN
        :return:
        '''
        seq_id2batch_id = [[seq.batch_id for seq in sequence_list.extract()] for sequence_list in sequence_lists]

        # to easily map the partial_sequences back to the flattened_sequences
        seq_id = 0
        flattened_id_map = []
        for sequence_list in sequence_lists:
            seq_ids = []
            for seq in sequence_list.extract():
                seq_ids.append(seq_id)
                seq_id += 1
            flattened_id_map.append(seq_ids)

        flattened_sequences = list(itertools.chain(*[seq.extract() for seq in sequence_lists]))
        batch_size = len(flattened_sequences)

        # if it's oov, replace it with <unk> (batch_size, 1)
        inputs = torch.cat([Variable(torch.LongTensor([seq.sentence[-1]] if seq.sentence[-1] < self.model.vocab_size else [self.model.unk_word])) for seq in flattened_sequences]).view(batch_size, -1)
        
        # (batch_size, trg_hidden_dim)
        dec_hiddens = torch.cat([seq.dec_hidden for seq in flattened_sequences]).view(1,batch_size,-1)
        
        contexts = torch.cat([seq.context for seq in flattened_sequences]).view(batch_size, *flattened_sequences[0].context.size())
        ctx_mask = torch.cat([seq.ctx_mask for seq in flattened_sequences]).view(batch_size, *flattened_sequences[0].ctx_mask.size())
        src_oovs = torch.cat([seq.src_oov for seq in flattened_sequences]).view(batch_size, *flattened_sequences[0].src_oov.size())
        oov_lists = [seq.oov_list for seq in flattened_sequences]

        
        if torch.cuda.is_available() and self.opt.use_gpu:
            inputs = inputs.cuda()
            if isinstance(flattened_sequences[0].dec_hidden, tuple):
                dec_hiddens = (dec_hiddens[0].cuda(), dec_hiddens[1].cuda())
            else:
                dec_hiddens = dec_hiddens.cuda()
            contexts = contexts.cuda()
            ctx_mask = ctx_mask.cuda()
            src_oovs = src_oovs.cuda()

        return seq_id2batch_id, flattened_id_map, inputs, dec_hiddens, contexts, ctx_mask, src_oovs, oov_lists

    def beam_search(self, src_input, src_len, src_oov, oov_list, word2id):
        """Runs beam search sequence generation given input (padded word indexes)

        Args:
          initial_input: An initial input for the model -
                         list of batch size holding the first input for every entry.
        Returns:
          A list of batch size, each the most likely sequence from the possible beam_size candidates.
        """
        self.model.eval()
        batch_size = len(src_input)
        src_mask = self.get_mask(src_input)  # same size as input_src
        src_context, src_h = self.model.encode(src_input, src_len)

        if self.model.decoder_name == 'Memory Network':
            # self.model.Memory_Decoder.set_ntm(src_context,src_mask,self.model.trg_hidden_dim)
            self.model.Memory_Decoder.ntm.set_memory(src_context,src_mask)
        
        dec_hiddens = src_h
        
        # print(src_context.size(),src_h.size())
        # each dec_hidden is (trg_seq_len, dec_hidden_dim)
        initial_input = [word2id[pykp.io.BOS_WORD]] * batch_size #[BOS_WORD] 
        dec_hiddens = [dec_hiddens[:,i,:] for i in range(batch_size)]

        partial_sequences = [TopN_heap(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN_heap(sys.maxsize) for _ in range(batch_size)]

        for batch_i in range(batch_size):
            seq = Sequence(
                batch_id=batch_i,
                sentence=[initial_input[batch_i]],
                dec_hidden=dec_hiddens[batch_i],
                context=src_context[batch_i],
                ctx_mask=src_mask[batch_i],
                src_oov=src_oov[batch_i],
                oov_list=oov_list[batch_i],
                logprobs=[],
                score=0.0,
                attention=[])
            partial_sequences[batch_i].push(seq)

        num_partial_sequences = sum([len(batch_seqs) for batch_seqs in partial_sequences])    
        
        '''
        Run beam search.
        '''
        for current_len in range(1, self.max_sequence_length + 1):

            # print('partial length',len(partial_sequences))
            # the total number of partial sequences of all the batches
            num_partial_sequences = sum([len(batch_seqs) for batch_seqs in partial_sequences])
            # print(num_partial_sequences,'num_partial_sequences')
            if num_partial_sequences == 0:
                # We have run out of partial candidates; often happens when beam_size is small
                break

            # flatten 2d sequences (batch_size, beam_size) into 1d batches (batch_size * beam_size) to feed model
            seq_id2batch_id, flattened_id_map, inputs, dec_hiddens, contexts, ctx_mask, src_oovs, oov_lists = self.sequence_to_batch(partial_sequences)
            
            # Run one-step generation. probs=(batch_size, 1, K), dec_hidden=tuple of (1, batch_size, trg_hidden_dim)

            if self.model.decoder_name == 'Memory Network':
                
                self.model.Memory_Decoder.ntm.set_memory(contexts,ctx_mask)

                log_probs,new_dec_hiddens,attn_weights = self.model.memory_generate(
                trg_inputs = inputs,
                dec_hidden = dec_hiddens,
                encoder_outputs = contexts,
                src_map=src_oovs,
                oov_lists=oov_lists,
                ctx_mask=ctx_mask,
                max_len=1,
                )
            else:
                log_probs, new_dec_hiddens, attn_weights = self.model.generate(
                    trg_input=inputs,
                    dec_hidden=dec_hiddens,
                    enc_context=contexts,
                    ctx_mask=ctx_mask,
                    src_map=src_oovs,
                    oov_list=oov_lists,
                    # k           =self.beam_size+1,
                    max_len=1,
                    return_attention=self.return_attention
                )
            
            # squeeze these outputs, (hyp_seq_size, trg_len=1, K+1) -> (hyp_seq_size, K+1)
            probs, words = log_probs.data.topk(self.beam_size, dim=-1)
            words = words.squeeze(1)
            probs = probs.squeeze(1)
            # (hyp_seq_size, trg_len=1, src_len) -> (hyp_seq_size, src_len)
            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
            else:
                attn_weights = attn_weights.squeeze(1)

            # tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, hyp_seq_size, trg_hidden_dim), squeeze the first dim
            if isinstance(new_dec_hiddens, tuple):
                new_dec_hiddens1 = new_dec_hiddens[0].squeeze(0)
                new_dec_hiddens2 = new_dec_hiddens[1].squeeze(0)
                new_dec_hiddens = [(new_dec_hiddens1[i], new_dec_hiddens2[i]) for i in range(num_partial_sequences)]
            else:
                
                new_dec_hiddens = new_dec_hiddens.squeeze(0)
                new_dec_hiddens = [new_dec_hiddens[i]    for i in range(num_partial_sequences)]
            
            # For every partial_sequence (num_partial_sequences in total), find and trim to the best hypotheses (beam_size in total)
            for batch_i in range(batch_size):
                num_new_hyp_in_batch = 0
                new_partial_sequences = TopN_heap(self.beam_size)

                for partial_id, partial_seq in enumerate(partial_sequences[batch_i].extract()):
                    num_new_hyp = 0
                    flattened_seq_id = flattened_id_map[batch_i][partial_id]

                    # check each new beam and decide to add to hypotheses or completed list
                    for beam_i in range(self.beam_size):
                        w = words[flattened_seq_id][beam_i]
                        

                        # score=0 means this is the first word <BOS>, empty the sentence
                        if partial_seq.score != 0:
                            new_sent = copy.copy(partial_seq.sentence)
                        else:
                            new_sent = []
                        
                        # if w has appeared before, ignore current hypothese
                        if w in new_sent:
                            continue

                        # print(new_sent,w,w in new_sent)
                        new_sent.append(w)


                        # if w >= 50000 and len(partial_seq.oov_list)==0:
                        #     print(new_sent)
                        #     print(partial_seq.oov_list)
                        #     pass

                        new_partial_seq = Sequence(
                            batch_id=partial_seq.batch_id,
                            sentence=new_sent,
                            dec_hidden=None,
                            context=partial_seq.context,
                            ctx_mask=partial_seq.ctx_mask,
                            src_oov=partial_seq.src_oov,
                            oov_list=partial_seq.oov_list,
                            logprobs=copy.copy(partial_seq.logprobs),
                            score=copy.copy(partial_seq.score),
                            attention=copy.copy(partial_seq.attention)
                        )

                        # we have generated self.beam_size new hypotheses for current hyp, stop generating
                        if num_new_hyp >= self.beam_size:
                            break

                        # dec_hidden and attention of this partial_seq are shared by its descendant beams
                        new_partial_seq.dec_hidden = new_dec_hiddens[flattened_seq_id]
                        # print(new_partial_seq.dec_hidden.size(),'new_dec_hiddens')
                        if self.return_attention:
                            if isinstance(attn_weights, tuple):  # if it's (attn, copy_attn)
                                attn_weights = (attn_weights[0].squeeze(1), attn_weights[1].squeeze(1))
                                new_partial_seq.attention.append((attn_weights[0][flattened_seq_id], attn_weights[1][flattened_seq_id]))
                            else:
                                new_partial_seq.attention.append(attn_weights[flattened_seq_id])
                        else:
                            new_partial_seq.attention = None

                        new_partial_seq.logprobs.append(probs[flattened_seq_id][beam_i])
                        new_partial_seq.score = new_partial_seq.score + probs[flattened_seq_id][beam_i]

                        # if predict EOS, push it into complete_sequences
                        if w == self.eos_id:
                            
                            if self.length_normalization_factor > 0:
                                L = self.length_normalization_const
                                length_penalty = (L + len(new_partial_seq.sentence)) / (L + 1)
                                # new_partial_seq.score /= len(new_partial_seq.sentence)
                                # print(new_partial_seq.sentence,len(new_partial_seq.sentence))
                                new_partial_seq.score /= length_penalty ** self.length_normalization_factor
                            # print(new_partial_seq.sentence)

                            if len(new_partial_seq.sentence) != len(set(new_partial_seq.sentence)):
                                print(new_partial_seq.sentence)

                            complete_sequences[new_partial_seq.batch_id].push(new_partial_seq)
                        else:
                            
                            new_partial_sequences.push(new_partial_seq)
                            
                            num_new_hyp += 1
                            num_new_hyp_in_batch += 1

                partial_sequences[batch_i] = new_partial_sequences
                
                # print('Batch=%d, \t#(hypothese) = %d, \t#(completed) = %d \t #(new_hyp_explored)=%d' % (batch_i, len(partial_sequences[batch_i]), len(complete_sequences[batch_i]), num_new_hyp_in_batch))
                
              
        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.

        # append all the partial_sequences to complete
        # [complete_sequences[s.batch_id] for s in partial_sequences]
        for batch_i in range(batch_size):
            if len(complete_sequences[batch_i]) == 0:
                complete_sequences[batch_i] = partial_sequences[batch_i]
            complete_sequences[batch_i] = complete_sequences[batch_i].extract(sort=True,topk=100)

        # print(complete_sequences[0][0])
        return complete_sequences

    