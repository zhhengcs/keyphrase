# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
import torch.nn.functional as F

import pykp
from pykp.eric_layers import GetMask, masked_softmax, TimeDistributedDense
from attention_layers import Google_self_attention,Cross_attention

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import time



class Attention(nn.Module):
    def __init__(self, enc_dim, trg_dim, method='general'):
        super(Attention, self).__init__()
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(enc_dim, trg_dim)
        elif self.method == 'concat':
            attn = nn.Linear(enc_dim + trg_dim, trg_dim)
            v = nn.Linear(trg_dim, 1)
            self.attn = TimeDistributedDense(mlp=attn)
            self.v = TimeDistributedDense(mlp=v)

        self.softmax = nn.Softmax()

        # input size is enc_dim + trg_dim as it's a concatenation of both context vectors and target hidden state
        # for Dot Attention, context vector has been converted to trg_dim first

        if self.method == 'dot':
            self.linear_out = nn.Linear(2 * trg_dim, trg_dim, bias=False)  # the W_c in Eq. 5 Luong et al. 2016 [Effective Approaches to Attention-based Neural Machine Translation]
        else:
            self.linear_out = nn.Linear(enc_dim + trg_dim, trg_dim, bias=False)  # the W_c in Eq. 5 Luong et al. 2016 [Effective Approaches to Attention-based Neural Machine Translation]

        self.tanh = nn.Tanh()

    def score(self, hiddens, encoder_outputs, encoder_mask=None):
        '''
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        if self.method == 'dot':
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            energies = torch.bmm(hiddens, encoder_outputs.transpose(1, 2))  # (batch, trg_len, src_len)
        elif self.method == 'general':
            # print(encoder_outputs.type(),encoder_outputs.size())
            energies = self.attn(encoder_outputs)  # (batch, src_len, trg_hidden_dim)
            if encoder_mask is not None:
                energies =  energies * encoder_mask.view(encoder_mask.size(0), encoder_mask.size(1), 1)
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            
            # print(hiddens.size(),'hidden')
            # print(energies.size(),'energies')
            energies = torch.bmm(hiddens, energies.transpose(1,2))  # (batch, trg_len, src_len)
        
        elif self.method == 'concat':
            energies = []
            batch_size = encoder_outputs.size(0)
            src_len = encoder_outputs.size(1)
            for i in range(hiddens.size(1)):
                hidden_i = hiddens[:, i: i + 1, :].expand(-1, src_len, -1)  # (batch, src_len, trg_hidden_dim)
                concated = torch.cat((hidden_i, encoder_outputs), 2)  # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
                if encoder_mask is not None:
                    concated =  concated * encoder_mask.view(encoder_mask.size(0), encoder_mask.size(1), 1)  # (batch_size, src_len, dec_hidden_dim + enc_hidden_dim)
                energy = self.tanh(self.attn(concated, encoder_mask))  # (batch_size, src_len, dec_hidden_dim)
                if encoder_mask is not None:
                    energy =  energy * encoder_mask.view(encoder_mask.size(0), encoder_mask.size(1), 1)  # (batch_size, src_len, dec_hidden_dim)
                energy = self.v(energy, encoder_mask).squeeze(-1)  # (batch_size, src_len)
                energies.append(energy)
            energies = torch.stack(energies, dim=1)  # (batch_size, trg_len, src_len)
            if encoder_mask is not None:
                energies =  energies * encoder_mask.view(encoder_mask.size(0), 1, encoder_mask.size(1))

        return energies.contiguous()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        '''
        Compute the attention and h_tilde, inputs/outputs must be batch first
        param:
            hidden: (batch_size, trg_len, trg_hidden_dim)
            encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
      
        return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        '''
        """
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1))) # src_seq_len * batch_size
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(encoder_outputs.size(0)):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, transpose to (batch_size * src_seq_len)
        attn = torch.nn.functional.softmax(attn_energies.t())
        # get the weighted context, (batch_size, src_layer_number * src_encoder_dim)
        weighted_context = torch.bmm(encoder_outputs.permute(1, 2, 0), attn.unsqueeze(2)).squeeze(2)  # (batch_size, src_hidden_dim * num_directions)
        """

        batch_size = hidden.size(0)
        src_len = encoder_outputs.size(1)
        trg_len = hidden.size(1)
        context_dim = encoder_outputs.size(2)
        trg_hidden_dim = hidden.size(2)

        # hidden (batch_size, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, with consideration of masks
        if encoder_mask is None:
            attn_weights = torch.nn.functional.softmax(attn_energies.view(-1, src_len), dim=1).view(batch_size, trg_len, src_len)  # (batch_size, trg_len, src_len)
        else:
            attn_energies = attn_energies * encoder_mask.view(encoder_mask.size(0), 1, encoder_mask.size(1))  # (batch, trg_len, src_len)
            attn_weights = masked_softmax(attn_energies, encoder_mask.view(encoder_mask.size(0), 1, encoder_mask.size(1)), -1)  # (batch_size, trg_len, src_len)

        # reweighting context, attn (batch_size, trg_len, src_len) * encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size, trg_len, src_hidden_dim)
        # print(attn_weights.size(),'attn weight size')
        # print(encoder_outputs.size(),'encoder output size')
        weighted_context = torch.bmm(attn_weights, encoder_outputs)

        # get h_tilde by = tanh(W_c[c_t, h_t]), both hidden and h_tilde are (batch_size, trg_hidden_dim)
        # (batch_size, trg_len=1, src_hidden_dim + trg_hidden_dim)
        h_tilde = torch.cat((weighted_context, hidden), 2)
        # (batch_size * trg_len, src_hidden_dim + trg_hidden_dim) -> (batch_size * trg_len, trg_hidden_dim)
        h_tilde = self.tanh(self.linear_out(h_tilde.view(-1, context_dim + trg_hidden_dim)))

        # return h_tilde (batch_size, trg_len, trg_hidden_dim), attn (batch_size, trg_len, src_len) and energies (before softmax)
        return h_tilde.view(batch_size, trg_len, trg_hidden_dim), attn_weights, attn_energies

    def forward_(self, hidden, context):
        """
        Original forward for DotAttention, it doesn't work if the dim of encoder and decoder are not same
        input and context must be in same dim: return Softmax(hidden.dot([c for c in context]))
        input: batch x hidden_dim
        context: batch x source_len x hidden_dim
        """
        # start_time = time.time()
        target = self.linear_in(hidden).unsqueeze(2)  # batch x hidden_dim x 1
        # print("---target set  %s seconds ---" % (time.time() - start_time))

        # Get attention, size=(batch_size, source_len, 1) -> (batch_size, source_len)
        attn = torch.bmm(context, target).squeeze(2)  # batch x source_len
        # print("--attenstion - %s seconds ---" % (time.time() - start_time))

        attn = self.softmax(attn)
        # print("---attn softmax  %s seconds ---" % (time.time() - start_time))

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch_size x 1 x source_len
        # print("---attn view %s seconds ---" % (time.time() - start_time))

        # Get the weighted context vector
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch_size x hidden_dim
        # print("---weighted context %s seconds ---" % (time.time() - start_time))

        # Update h by tanh(torch.cat(weighted_context, input))
        h_tilde = torch.cat((weighted_context, hidden), 1)  # batch_size * (src_hidden_dim + trg_hidden_dim)
        h_tilde = self.tanh(self.linear_out(h_tilde))  # batch_size * trg_hidden_dim
        # print("--- %s seconds ---" % (time.time() - start_time))

        return h_tilde, attn


class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt,encoder_name=None,decoder_name=None):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.src_hidden_dim = opt.rnn_size
        self.trg_hidden_dim = opt.rnn_size
        self.ctx_hidden_dim = opt.rnn_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.nlayers_src = opt.enc_layers
        self.nlayers_trg = opt.dec_layers
        self.dropout = opt.dropout

        self.max_trg_seq_length = opt.max_trg_seq_length

        self.pad_token_src = opt.word2id[pykp.io.PAD_WORD]
        self.pad_token_trg = opt.word2id[pykp.io.PAD_WORD]
        self.unk_word = opt.word2id[pykp.io.UNK_WORD]

        self.attention_mode = opt.attention_mode    # 'dot', 'general', 'concat'
        self.input_feeding = opt.input_feeding

        self.copy_attention = opt.copy_attention    # bool, enable copy attention or not
        self.copy_mode = opt.copy_mode         # same to `attention_mode`
        self.copy_input_feeding = opt.copy_input_feeding
        self.reuse_copy_attn = opt.reuse_copy_attn
        self.copy_gate = opt.copy_gate

        self.must_teacher_forcing = opt.must_teacher_forcing
        self.teacher_forcing_ratio = opt.teacher_forcing_ratio
        self.scheduled_sampling = opt.scheduled_sampling
        self.scheduled_sampling_batches = opt.scheduled_sampling_batches
        self.scheduled_sampling_type = 'inverse_sigmoid'  # decay curve type: linear or inverse_sigmoid
        self.current_batch = 0  # for scheduled sampling


        self.use_gpu = opt.use_gpu
        self.encoder_name = 'BiGRU' # BiGRU or Attention
        self.decoder_name = 'Memory Network' #CopyRNN or Memory Network

        if self.scheduled_sampling:
            logging.info("Applying scheduled sampling with %s decay for the first %d batches" % (self.scheduled_sampling_type, self.scheduled_sampling_batches))
        if self.must_teacher_forcing or self.teacher_forcing_ratio >= 1:
            logging.info("Training with All Teacher Forcing")
        elif self.teacher_forcing_ratio <= 0:
            logging.info("Training with All Sampling")
        else:
            logging.info("Training with Teacher Forcing with static rate=%f" % self.teacher_forcing_ratio)

        self.get_mask = GetMask(self.pad_token_src)

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_token_src
        )

        self.encoder = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.src_hidden_dim,
            num_layers=self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.trg_hidden_dim,
            num_layers=self.nlayers_trg,
            bidirectional=False,
            batch_first=False,
            dropout=self.dropout
        )

        self.attention_layer = Attention(self.src_hidden_dim * self.num_directions, self.trg_hidden_dim, method=self.attention_mode)

        self.encoder2decoder_hidden = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        self.encoder2decoder_cell = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)

        # copy attention
        if self.copy_attention:
            if self.copy_mode == None and self.attention_mode:
                self.copy_mode = self.attention_mode
            assert self.copy_mode != None
            assert self.unk_word != None
            logging.info("Applying Copy Mechanism, type=%s" % self.copy_mode)
            # for Gu's model
            self.copy_attention_layer = Attention(self.src_hidden_dim * self.num_directions, self.trg_hidden_dim, method=self.copy_mode)
            # for See's model
            # self.copy_gate            = nn.Linear(self.trg_hidden_dim, self.vocab_size)
        else:
            self.copy_mode = None
            self.copy_input_feeding = False
            self.copy_attention_layer = None

        # setup for input-feeding, add a bridge to compress the additional inputs. Note that input-feeding cannot work with teacher-forcing
        self.dec_input_dim = self.emb_dim  # only input the previous word
        if self.input_feeding:
            logging.info("Applying input feeding")
            self.dec_input_dim += self.trg_hidden_dim
        if self.copy_input_feeding:
            logging.info("Applying copy input feeding")
            self.dec_input_dim += self.trg_hidden_dim
        if self.dec_input_dim == self.emb_dim:
            self.dec_input_bridge = None
        else:
            self.dec_input_bridge = nn.Linear(self.dec_input_dim, self.emb_dim)

        self.init_weights()

        if self.encoder_name=='Attention':
            
            self.slf_attn_layer = Google_self_attention(opt,add_pos = False)

            self.cross_attn_layer = Cross_attention(opt)
        # elif self.encoder_name=='BiGRU':
        #     self. 

        if self.decoder_name == 'Memory Network':
            self.Memory_Decoder = MeMDecoder(opt)
            self.Memory_Decoder.embedding.weight = self.embedding.weight
            

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        self.encoder2decoder_hidden.bias.data.fill_(0)
        self.encoder2decoder_cell.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def init_encoder_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        if torch.cuda.is_available() and self.use_gpu:
            return h0_encoder.cuda()#, c0_encoder.cuda()

        return h0_encoder#, c0_encoder

    def init_decoder_state(self, enc_h, enc_c=None):
        # prepare the init hidden vector for decoder, (batch_size, num_layers * num_directions * enc_hidden_dim) -> (num_layers * num_directions, batch_size, dec_hidden_dim)
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder_hidden(enc_h)).unsqueeze(0)
        decoder_init_cell = nn.Tanh()(self.encoder2decoder_cell(enc_c)).unsqueeze(0)

        return decoder_init_hidden, decoder_init_cell

    def forward(self, input_src, input_src_len, input_trg, input_src_ext, oov_lists, 
                    trg_mask=None, ctx_mask=None,query_src = None):
        
        '''
        The differences of copy model from normal seq2seq here are:
         1. The size of decoder_logits is (batch_size, trg_seq_len, vocab_size + max_oov_number).Usually vocab_size=50000 and max_oov_number=1000. And only very few of (it's very rare to have many unk words, in most cases it's because the text is not in English)
         2. Return the copy_attn_weights as well. If it's See's model, the weights are same to attn_weights as it reuse the original attention
         3. Very important: as we need to merge probs of copying and generative part, thus we have to operate with probs instead of logits. Thus here we return the probs not logits. Respectively, the loss criterion outside is NLLLoss but not CrossEntropyLoss any more.
        :param
            input_src : numericalized source text, oov words have been replaced with <unk>
            input_trg : numericalized target text, oov words have been replaced with temporary oov index
            input_src_ext : numericalized source text in extended vocab, oov words have been replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words
        :returns
            decoder_logits      : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''
        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)  # same size as input_src

        
        if self.encoder_name == 'Attention':
            # encoder_outputs,encoder_hidden,query_output = self.attention_encoder(input_src,input_src_len,query_src,query_len)
            encoder_outputs,encoder_hidden = self.encode(input_src,input_src_len)
            
            # query_slf_mask = self.get_attn_padding_mask(query_src,query_src)
            # query_emb = self.embedding(query_src)
            # query_outputs,_ = self.slf_attn_layer(query_emb,None,query_slf_mask)

            # doc_mask = self.get_attn_padding_mask(query_src,input_src)
            # query_mask = self.get_attn_padding_mask(input_src,query_src)
            
            # encoder_outputs,encoder_hidden,query_outputs = self.cross_attn_layer(encoder_outputs,doc_mask,input_src_len,query_outputs,query_mask)
            
        elif self.encoder_name == 'BiGRU':
            encoder_outputs, encoder_hidden = self.encode(input_src, input_src_len)
            
        

        if self.decoder_name == 'Memory Network':
            
            memory_mask = self.get_mask(query_src)
            if self.encoder_name == 'Attention':
                # self.Memory_Decoder.ntm.set_memory(query_outputs,memory_mask)
                self.Memory_Decoder.ntm.set_memory(encoder_outputs,ctx_mask)
            elif self.encoder_name == 'BiGRU':
                self.Memory_Decoder.ntm.set_memory(encoder_outputs,ctx_mask)

            decoder_probs, decoder_hiddens, attn_weights, copy_attn_weights = self.memory_decode(trg_inputs=input_trg, 
                                                                    src_map=input_src_ext,
                                                                    oov_list=oov_lists, 
                                                                    encoder_outputs=encoder_outputs, 
                                                                    dec_hidden=encoder_hidden,
                                                                    trg_mask=trg_mask, 
                                                                    ctx_mask=ctx_mask,
                                                                    oov_lists=oov_lists
                                                                    )
        elif self.decoder_name == 'CopyRNN':
            decoder_probs, decoder_hiddens, attn_weights, copy_attn_weights = self.decode(trg_inputs=input_trg, 
                                                                            src_map=input_src_ext,
                                                                            oov_list=oov_lists, 
                                                                            enc_context=encoder_outputs, 
                                                                            enc_hidden=encoder_hidden,
                                                                            trg_mask=trg_mask, 
                                                                            ctx_mask=ctx_mask)

        return decoder_probs, decoder_hiddens, (attn_weights, copy_attn_weights)

    def memory_decode(self,trg_inputs,src_map,oov_list,encoder_outputs,dec_hidden,trg_mask,ctx_mask,oov_lists):
        

        s_t_1 = dec_hidden
        decoder_probs = []
        decoder_hiddens = []
        enc_batch_extend_vocab = src_map
        # maximum length to unroll, ignore the last word (must be padding)
        max_dec_len = trg_inputs.size(1) - 1
        batch_size,_ = trg_inputs.size()

        c_t_1 = Variable(torch.zeros(batch_size, 1, self.ctx_hidden_dim))
        
        if torch.cuda.is_available() and self.use_gpu:
            c_t_1 = c_t_1.cuda()
            
        
        max_art_oovs = max([len(x) for x in oov_lists])
        if max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
            if self.use_gpu:
                extra_zeros=extra_zeros.cuda()
        else:
            extra_zeros = None

        for di in range(max_dec_len):

            y_t_1 = trg_inputs[:, di]
            final_dist, s_t_1,  c_t_1, _, = self.Memory_Decoder(y_t_1, s_t_1,
                                                        encoder_outputs, ctx_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab)
            decoder_probs.append(final_dist)
            decoder_hiddens.append(s_t_1)
            # print('-----------------step %d--------------------'%di)

        decoder_probs = torch.stack(decoder_probs,dim=1)
        # decoder_hiddens = 
        # print(decoder_probs.size())        
        decoder_probs = torch.log(decoder_probs+1e-12)
        
        return decoder_probs,decoder_hiddens,None,None
    

    def memory_generate(self,trg_inputs,dec_hidden,encoder_outputs,ctx_mask,src_map,oov_lists,max_len=1,return_attention=False):
        s_t_1 = dec_hidden
        s_t_1 = dec_hidden
        decoder_probs = []
        decoder_hiddens = []
        enc_batch_extend_vocab = src_map

        batch_size,_ = trg_inputs.size()

        c_t_1 = Variable(torch.zeros(batch_size,1,self.ctx_hidden_dim))

        if torch.cuda.is_available() and self.use_gpu:
            c_t_1 = c_t_1.cuda()

        max_art_oovs = max([len(x) for x in oov_lists])
        if max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
            if self.use_gpu and torch.cuda.is_available():
                extra_zeros=extra_zeros.cuda()
        else:
            extra_zeros = None
      
        y_t_1 = trg_inputs.view(-1)
        # print(y_t_1.size(),'y_t_1')
        final_dist, s_t_1,  c_t_1, attn_weights, = self.Memory_Decoder(y_t_1, s_t_1,
                                                    encoder_outputs, ctx_mask, c_t_1,
                                                    extra_zeros, enc_batch_extend_vocab)
        final_dist = torch.log(final_dist)
        return final_dist,s_t_1,attn_weights
        
    def encode(self, input_src, input_src_len):
        """
        Propogate input through the network.
        """
        # initial encoder state, two zero-matrix as h and c at time=0
        self.h_encoder = self.init_encoder_state(input_src)  # (self.encoder.num_layers * self.num_directions, batch_size, self.src_hidden_dim)

        # input (batch_size, src_len), src_emb (batch_size, src_len, emb_dim)
        
        src_emb = self.embedding(input_src)

        src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len, batch_first=True)

        src_h, src_state = self.encoder(src_emb, self.h_encoder)
        src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True)
        
        return src_h, src_state
        
    # def attention_encoder(self,input_src,input_src_len,query_src,query_len):
    #     input_mask = self.get_attn_padding_mask(input_src,input_src)
    #     src_emb = self.embedding(input_src)
        
    #     enc_output,_ =  self.slf_attn_layer(src_emb,None,input_mask)




    def get_attn_padding_mask(self,seq_q, seq_k):
        ''' Indicate the padding-related part to mask '''
        assert seq_q.dim() == 2 and seq_k.dim() == 2
        mb_size, len_q = seq_q.size()
        mb_size, len_k = seq_k.size()

        pad_attn_mask = seq_k.data.eq(self.pad_token_src).unsqueeze(1)   # bx1xsk
        pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
        if torch.cuda.is_available() and self.use_gpu:
            return pad_attn_mask.cuda()
        else:
            return pad_attn_mask

    def merge_decode_inputs(self, trg_emb, h_tilde, copy_h_tilde):
        '''
        Input-feeding: merge the information of current word and attentional hidden vectors
        :param trg_emb: (batch_size, 1, embed_dim)
        :param h_tilde: (batch_size, 1, trg_hidden)
        :param copy_h_tilde: (batch_size, 1, trg_hidden)
        :return:
        '''
        trg_emb = trg_emb.permute(1, 0, 2)  # (1, batch_size, embed_dim)
        inputs = trg_emb
        if self.input_feeding:
            h_tilde = h_tilde.permute(1, 0, 2)  # (1, batch_size, trg_hidden)
            inputs = torch.cat((inputs, h_tilde), 2)  # (1, batch_size, inputs_dim+trg_hidden)
        if self.copy_input_feeding:
            copy_h_tilde = copy_h_tilde.permute(1, 0, 2)  # (1, batch_size, inputs_dim+trg_hidden)
            inputs = torch.cat((inputs, copy_h_tilde), 2)

        if self.dec_input_bridge:
            dec_input = nn.Tanh()(self.dec_input_bridge(inputs))
        else:
            dec_input = trg_emb

        # if isinstance(dec_hidden, tuple):
        #     dec_hidden = (h_tilde.permute(1, 0, 2), dec_hidden[1])
        # else:
        #     dec_hidden = h_tilde.permute(1, 0, 2)
        # trg_input = trg_inputs[:, di + 1].unsqueeze(1)

        return dec_input

    def decode(self, trg_inputs, src_map, oov_list, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        :param
                trg_input:         (batch_size, trg_len)
                src_map  :         (batch_size, src_len), almost the same with src but oov words are replaced with temporary oov 
                                    index, for copy mechanism to map the probs of pointed words to vocab words. 
                                    The word index can be beyond vocab_size, e.g. 50000, 50001, 50002 etc, 
                                    depends on how many oov words appear in the source text
                context vector:    (batch_size, src_len, hidden_size * num_direction) the outputs (hidden vectors) of encoder
                context mask:      (batch_size, src_len)
        :returns
            decoder_probs       : (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''
        batch_size = trg_inputs.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_inputs.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        # init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])
        init_hidden = enc_hidden

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)
            enc_context = enc_context * ctx_mask.view(ctx_mask.size() + (1,))

        # maximum length to unroll, ignore the last word (must be padding)
        max_length = trg_inputs.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        # because sequence-wise training is not compatible with input-feeding, so discard it
        # TODO 20180722, do_word_wisely_training=True is buggy
        do_word_wisely_training = False
        if not do_word_wisely_training:
            '''
            Teacher Forcing
            (1) Feedforwarding RNN
            '''
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_inputs = trg_inputs[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_inputs)  # (batch_size, trg_len, embed_dim)
            trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, dec_hidden = self.decoder(
                trg_emb, init_hidden
            )
            '''
            (2) Standard Attention
            '''
            # Get the h_tilde (batch_size, trg_len, trg_hidden_dim) and attention weights (batch_size, trg_len, src_len)
            h_tildes, attn_weights, attn_logits = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde), (batch_size, trg_len, trg_hidden_size) -> (batch_size * trg_len, vocab_size)
            # h_tildes=(batch_size, trg_len, trg_hidden_size) -> decoder2vocab(h_tildes.view)=(batch_size * trg_len, vocab_size) -> decoder_logits=(batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim)).view(batch_size, max_length, -1)

            '''
            (3) Copy Attention
            '''
            if self.copy_attention:
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if not self.reuse_copy_attn:
                    _, copy_weights, copy_logits = self.copy_attention_layer(decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
                else:
                    copy_logits = attn_logits

                # merge the generative and copying probs, (batch_size, trg_len, vocab_size + max_oov_number)
                decoder_log_probs = self.merge_copy_probs(decoder_logits, copy_logits, src_map, oov_list)  # (batch_size, trg_len, vocab_size + max_oov_number)
                decoder_outputs = decoder_outputs.permute(1, 0, 2)  # (batch_size, trg_len, trg_hidden_dim)
            else:
                decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits, dim=-1).view(batch_size, -1, self.vocab_size)

        else:
            '''
            Word Sampling
            (1) Feedforwarding RNN
            '''
            # take the first word (should be BOS <s>) of each target sequence (batch_size, 1)
            trg_input = trg_inputs[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs = []
            attn_weights = []
            copy_weights = []
            dec_hidden = init_hidden
            h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() and self.use_gpu else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
            copy_h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() and self.use_gpu else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))

            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input)  # (batch_size, 1, embed_dim)

                # input-feeding, attentional vectors h˜t are concatenated with inputs at the next time steps
                dec_input = self.merge_decode_inputs(trg_emb, h_tilde, copy_h_tilde)

                # run RNN decoder with inputs (trg_len first)
                decoder_output, dec_hidden = self.decoder(
                    dec_input, dec_hidden
                )

                '''
                (2) Standard Attention
                '''
                # Get the h_tilde (hidden after attention) and attention weights. h_tilde (batch_size,1,trg_hidden), attn_weight & attn_logit(batch_size,1,src_len)
                h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # h_tilde=(batch_size, 1, trg_hidden_size) -> decoder2vocab(h_tilde.view)=(batch_size * 1, vocab_size) -> decoder_logit=(batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim)).view(batch_size, 1, -1)

                '''
                (3) Copy Attention
                '''
                if self.copy_attention:
                    # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                    if not self.reuse_copy_attn:
                        copy_h_tilde, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
                    else:
                        copy_h_tilde, copy_weight, copy_logit = h_tilde, attn_weight, attn_logit

                    # merge the generative and copying probs (batch_size, 1, vocab_size + max_oov_number)
                    decoder_log_prob = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)
                else:
                    decoder_log_prob = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, -1, self.vocab_size)
                    copy_weight = None

                '''
                Prepare for the next iteration
                '''
                # prepare the next input word
                if self.do_teacher_forcing():
                    # truncate the last word, as there's no further word after it for decoder to predict
                    trg_input = trg_inputs[:, di + 1].unsqueeze(1)
                else:
                    # find the top 1 predicted word
                    top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                    # if it's a oov, replace it with <unk>
                    top_idx[top_idx >= self.vocab_size] = self.unk_word
                    top_idx = Variable(top_idx.squeeze(2))
                    # top_idx and next_index are (batch_size, 1)
                    trg_input = top_idx.cuda() if torch.cuda.is_available() and self.use_gpu else top_idx

                # Save results of current step. Permute to trg_len first, otherwise the cat operation would mess up things
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                if self.copy_attention:
                    copy_weights.append(copy_weight.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs = torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, hidden_size)
            attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)
            if self.copy_attention:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs (logits after log_softmax), hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights, copy_weights

    def merge_oov2unk(self, decoder_log_prob, max_oov_number):
        '''
        Merge the probs of oov words to the probs of <unk>, in order to generate the next word
        :param decoder_log_prob: log_probs after merging generative and copying (batch_size, trg_seq_len, vocab_size + max_oov_number)
        :return:
        '''
        batch_size, seq_len, _ = decoder_log_prob.size()
        # range(0, vocab_size)
        vocab_index = Variable(torch.arange(start=0, end=self.vocab_size).type(torch.LongTensor))
        # range(vocab_size, vocab_size+max_oov_number)
        oov_index = Variable(torch.arange(start=self.vocab_size, end=self.vocab_size + max_oov_number).type(torch.LongTensor))
        oov2unk_index = Variable(torch.zeros(batch_size * seq_len, max_oov_number).type(torch.LongTensor) + self.unk_word)

        if torch.cuda.is_available() and self.use_gpu:
            vocab_index = vocab_index.cuda()
            oov_index = oov_index.cuda()
            oov2unk_index = oov2unk_index.cuda()

        merged_log_prob = torch.index_select(decoder_log_prob, dim=2, index=vocab_index).view(batch_size * seq_len, self.vocab_size)
        oov_log_prob = torch.index_select(decoder_log_prob, dim=2, index=oov_index).view(batch_size * seq_len, max_oov_number)

        # all positions are zeros except the index of unk_word, then add all the probs of oovs to <unk>
        merged_log_prob = merged_log_prob.scatter_add_(1, oov2unk_index, oov_log_prob)
        merged_log_prob = merged_log_prob.view(batch_size, seq_len, self.vocab_size)

        return merged_log_prob

    def merge_copy_probs(self, decoder_logits, copy_logits, src_map, oov_list):
        '''
        The function takes logits as inputs here because Gu's model applies softmax in the end, to normalize generative/copying together
        The tricky part is, Gu's model merges the logits of generative and copying part instead of probabilities,
            then simply initialize the entended part to zeros would be erroneous because many logits are large negative floats.
        To the sentences that have oovs it's fine. But if some sentences in a batch don't have oovs but mixed with sentences have oovs, the extended oov part would be ranked highly after softmax (zero is larger than other negative values in logits).
        Thus we have to carefully initialize the oov-extended part of no-oov sentences to negative infinite floats.
        Note that it may cause exception on early versions like on '0.3.1.post2', but it works well on 0.4 ({RuntimeError}in-place operations can be only used on variables that don't share storage with any other variables, but detected that there are 2 objects sharing it)
        :param decoder_logits: (batch_size, trg_seq_len, vocab_size)
        :param copy_logits:    (batch_size, trg_len, src_len) the pointing/copying logits of each target words
        :param src_map:        (batch_size, src_len)
        :return:
            decoder_copy_probs: return the log_probs (batch_size, trg_seq_len, vocab_size + max_oov_number)
        '''
        batch_size, max_length, _ = decoder_logits.size()
        src_len = src_map.size(1)

        # set max_oov_number to be the max number of oov
        max_oov_number = max([len(oovs) for oovs in oov_list])

        # flatten and extend size of decoder_probs from (vocab_size) to (vocab_size+max_oov_number)
        flattened_decoder_logits = decoder_logits.view(batch_size * max_length, self.vocab_size)
        if max_oov_number > 0:
            '''
            extended_zeros           = Variable(torch.zeros(batch_size * max_length, max_oov_number))
            extended_zeros           = extended_zeros.cuda() if torch.cuda.is_available() else extended_zeros
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_zeros), dim=1)
            '''
            extended_logits = Variable(torch.FloatTensor([[0.0] * len(oov) + [float('-inf')] * (max_oov_number - len(oov)) for oov in oov_list]))
            extended_logits = extended_logits.unsqueeze(1).expand(batch_size, max_length, max_oov_number).contiguous().view(batch_size * max_length, -1)
            extended_logits = extended_logits.cuda() if torch.cuda.is_available() and self.use_gpu else extended_logits
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_logits), dim=1)

        # add probs of copied words by scatter_add_(dim, index, src), index should be in the same shape with src. decoder_probs=(batch_size * trg_len, vocab_size+max_oov_number), copy_weights=(batch_size, trg_len, src_len)
        expanded_src_map = src_map.unsqueeze(1).expand(batch_size, max_length, src_len).contiguous().view(batch_size * max_length, -1)  # (batch_size, src_len) -> (batch_size * trg_len, src_len)
        # flattened_decoder_logits.scatter_add_(dim=1, index=expanded_src_map, src=copy_logits.view(batch_size * max_length, -1))
        flattened_decoder_logits = flattened_decoder_logits.scatter_add_(1, expanded_src_map, copy_logits.view(batch_size * max_length, -1))

        # apply log softmax to normalize, ensuring it meets the properties of probability, (batch_size * trg_len, src_len)
        flattened_decoder_logits = torch.nn.functional.log_softmax(flattened_decoder_logits, dim=1)

        # reshape to batch first before returning (batch_size, trg_len, src_len)
        decoder_log_probs = flattened_decoder_logits.view(batch_size, max_length, self.vocab_size + max_oov_number)

        return decoder_log_probs

    def do_teacher_forcing(self):
        if self.scheduled_sampling:
            if self.scheduled_sampling_type == 'linear':
                teacher_forcing_ratio = 1 - float(self.current_batch) / self.scheduled_sampling_batches
            elif self.scheduled_sampling_type == 'inverse_sigmoid':
                # apply function k/(k+e^(x/k-m)), default k=1 and m=5, scale x to [0, 2*m], to ensure the many initial rounds are trained with teacher forcing
                x = float(self.current_batch) / self.scheduled_sampling_batches * 10 if self.scheduled_sampling_batches > 0 else 0.0
                teacher_forcing_ratio = 1. / (1. + np.exp(x - 5))
        elif self.must_teacher_forcing:
            teacher_forcing_ratio = 1.0
        else:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        # flip a coin
        coin = random.random()
        # logging.info('coin = %f, tf_ratio = %f' % (coin, teacher_forcing_ratio))

        do_tf = coin < teacher_forcing_ratio
        # if do_tf:
        #     logging.info("Training batches with Teacher Forcing")
        # else:
        #     logging.info("Training batches with All Sampling")

        return do_tf

    def generate(self, trg_input, dec_hidden, enc_context, ctx_mask=None, src_map=None, oov_list=None, max_len=1, return_attention=False):
        '''
        Given the initial input, state and the source contexts, return the top K restuls for each time step
        :param trg_input: just word indexes of target texts (usually zeros indicating BOS <s>)
        :param dec_hidden: hidden states for decoder RNN to start with
        :param enc_context: context encoding vectors
        :param src_map: required if it's copy model
        :param oov_list: required if it's copy model
        :param k (deprecated): Top K to return
        :param feed_all_timesteps: it's one-step predicting or feed all inputs to run through all the time steps
        :param get_attention: return attention vectors?
        :return:
        '''

        batch_size = trg_input.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_input.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim

        h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() and self.use_gpu else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
        copy_h_tilde = Variable(torch.zeros(batch_size, 1, trg_hidden_dim)).cuda() if torch.cuda.is_available() and self.use_gpu else Variable(torch.zeros(batch_size, 1, trg_hidden_dim))
        attn_weights = []
        copy_weights = []
        log_probs = []

        for i in range(max_len):
  
            trg_emb = self.embedding(trg_input)  # (batch_size, trg_len = 1, emb_dim)

            # Input-feeding, attentional vectors h˜t are concatenated with inputs at the next time steps
            dec_input = self.merge_decode_inputs(trg_emb, h_tilde, copy_h_tilde)

            # (seq_len, batch_size, hidden_size * num_directions)
            decoder_output, dec_hidden = self.decoder(
                dec_input, dec_hidden
            )
            # print(decoder_output.size(),'decoder_output size')
            # print(enc_context.size(),'enc context size')
            # Get the h_tilde (hidden after attention) and attention weights
            h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
            decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))

            if not self.copy_attention:
                decoder_log_prob = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
            else:
                decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if not self.reuse_copy_attn:
                    copy_h_tilde, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)
                else:
                    copy_h_tilde, copy_weight, copy_logit = h_tilde, attn_weight, attn_logit
                copy_weights.append(copy_weight.permute(1, 0, 2))  # (1, batch_size, src_len)
                # merge the generative and copying probs (batch_size, 1, vocab_size + max_unk_word)
                decoder_log_prob = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)

            # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
            top_1_v, top_1_idx = decoder_log_prob.data.topk(1, dim=-1)  # (batch_size, 1)
            trg_input = Variable(top_1_idx.squeeze(2))
            # trg_input           = Variable(top_1_idx).cuda() if torch.cuda.is_available() else Variable(top_1_idx) # (batch_size, 1)

            # append to return lists
            log_probs.append(decoder_log_prob.permute(1, 0, 2))  # (1, batch_size, vocab_size)
            attn_weights.append(attn_weight.permute(1, 0, 2))  # (1, batch_size, src_len)

        # permute to trg_len first, otherwise the cat operation would mess up things
        log_probs = torch.cat(log_probs, 0).permute(1, 0, 2)  # (batch_size, max_len, K)
        attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, max_len, src_seq_len)

        # Only return the hidden vectors of the last time step.
        #   tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, batch_size, trg_hidden_dim)

        # Return final outputs, hidden states, and attention weights (for visualization)
        if return_attention:
            if not self.copy_attention:
                return log_probs, dec_hidden, attn_weights
            else:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2)  # (batch_size, max_len, src_seq_len)
                return log_probs, dec_hidden, (attn_weights, copy_weights)
        else:
            return log_probs, dec_hidden

    def greedy_predict(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        src_h, (src_h_t, src_c_t) = self.encode(input_src)
        if torch.cuda.is_available() and self.use_gpu:
            input_trg = input_trg.cuda()
        decoder_logits, hiddens, attn_weights = self.decode_old(trg_input=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask, is_train=False)

        if torch.cuda.is_available():
            max_words_pred = decoder_logits.data.cpu().numpy().argmax(axis=-1).flatten()
        else:
            max_words_pred = decoder_logits.data.numpy().argmax(axis=-1).flatten()

        return max_words_pred

    def forward_without_copy(self, input_src, input_src_len, input_trg, trg_mask=None, ctx_mask=None):
        '''
        [Obsolete] To be compatible with the Copy Model, we change the output of logits to log_probs
        :param input_src: padded numeric source sequences
        :param input_src_len: (list of int) length of each sequence before padding (required for pack_padded_sequence)
        :param input_trg: padded numeric target sequences
        :param trg_mask:
        :param ctx_mask:

        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        '''
        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)  # same size as input_src
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_log_probs, decoder_hiddens, attn_weights = self.decode(trg_inputs=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_log_probs, decoder_hiddens, attn_weights

    def decode_without_copy(self, trg_inputs, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        [Obsolete] Initial decoder state h0 (batch_size, trg_hidden_size), converted from h_t of encoder (batch_size, src_hidden_size * num_directions) through a linear layer
            No transformation for cell state c_t. Pass directly to decoder.
            Nov. 11st: update: change to pass c_t as well
            People also do that directly feed the end hidden state of encoder and initialize cell state as zeros
        :param
                trg_input:         (batch_size, trg_len)
                context vector:    (batch_size, src_len, hidden_size * num_direction) is outputs of encoder
        :returns
            decoder_logits  : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs : (batch_size, trg_seq_len, hidden_size)
            attn_weights    : (batch_size, trg_seq_len, src_seq_len)
        '''
        batch_size = trg_inputs.size(0)
        src_len = enc_context.size(1)
        trg_len = trg_inputs.size(1)
        context_dim = enc_context.size(2)
        trg_hidden_dim = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        if self.attention_layer.method == 'dot':
            enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        # maximum length to unroll
        max_length = trg_inputs.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        if self.do_teacher_forcing():
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_inputs = trg_inputs[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_inputs)  # (batch_size, trg_len, embed_dim)
            trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, dec_hidden = self.decoder(
                trg_emb, init_hidden
            )
            # Get the h_tilde (hidden after attention) and attention weights, inputs/outputs must be batch first
            h_tildes, attn_weights, _ = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim))
            decoder_log_probs = torch.nn.functional.log_softmax(decoder_logits, dim=-1).view(batch_size, max_length, self.vocab_size)

            decoder_outputs = decoder_outputs.permute(1, 0, 2)

        else:
            # truncate the last word, as there's no further word after it for decoder to predict (batch_size, 1)
            trg_input = trg_inputs[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs = []
            attn_weights = []

            dec_hidden = init_hidden
            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input)  # (batch_size, trg_len, embed_dim)
                trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

                # input-feeding is not implemented

                # this is trg_len first
                decoder_output, dec_hidden = self.decoder(
                    trg_emb, dec_hidden
                )

                # Get the h_tilde (hidden after attention) and attention weights, both inputs and outputs are batch first
                h_tilde, attn_weight, _ = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context, encoder_mask=ctx_mask)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # (batch_size, trg_hidden_size) -> (batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))
                decoder_log_prob = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)

                '''
                Prepare for the next iteration
                '''
                # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
                top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                top_idx = Variable(top_idx.squeeze(2))
                # top_idx and next_index are (batch_size, 1)
                trg_input = top_idx.cuda() if torch.cuda.is_available() and self.use_gpu else top_idx

                # permute to trg_len first, otherwise the cat operation would mess up things
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            decoder_outputs = torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            attn_weights = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs, hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights


class Seq2SeqLSTMAttentionCascading(Seq2SeqLSTMAttention):
    def __init__(self, opt):
        super(Seq2SeqLSTMAttentionCascading, self).__init__(opt)

def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:

            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-0.1, 0.1)
            
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

class MeMDecoder(nn.Module):
    def __init__(self,opt):
        super(MeMDecoder, self).__init__()
        self.attention_network = Attention(opt.rnn_size,opt.rnn_size)
        self.embedding = nn.Embedding(opt.vocab_size, opt.word_vec_size,padding_idx=opt.word2id[pykp.io.PAD_WORD])
        self.x_context = nn.Linear(opt.rnn_size + opt.rnn_size, opt.rnn_size)
        self.s_context = nn.Linear(opt.rnn_size+opt.rnn_size,opt.rnn_size)

        self.gru1 = nn.GRU(opt.rnn_size,opt.rnn_size,num_layers=1,batch_first=True,bidirectional=False)
        self.gru2 = nn.GRU(opt.rnn_size,opt.rnn_size,num_layers=1,batch_first=True,bidirectional=False)
        init_rnn_wt(self.gru1)
        init_rnn_wt(self.gru2)
        # self.hidden_dim = opt.rnn_size

        self.out1 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.out2 = nn.Linear(opt.rnn_size, opt.vocab_size)
        self.ntm = NTMMemory(opt)
    
    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab):
        y_t_1_embd = self.embedding(y_t_1)
        
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd.unsqueeze(1)), -1))

        # first st
        gru_out, s_t_1_ = self.gru1(x, s_t_1)
        
        for i in range(2):
            context_t = self.ntm.Read(s_t_1_)
            _,s_t = self.gru2(context_t,s_t_1_)
            # s_t = self.s_context(s)
            self.ntm.Write(s_t)


        #End st
        hidden_dim = s_t.size()[-1]

        c_t, attn_dist,_ = self.attention_network(s_t.transpose(0,1), encoder_outputs, enc_padding_mask)
 
        output = torch.cat((s_t.view(-1,hidden_dim*1), c_t.view(-1,hidden_dim*1)), -1) # B x hidden_dim *2
        output = self.out1(output) # B x hidden_dim
        output = F.relu(output)
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=-1)

        vocab_dist_ = 0.5* vocab_dist
        attn_dist_ = (1 - 0.5) * attn_dist

        if extra_zeros is not None:    
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], -1)
        
        seq_len = attn_dist_.size()[-1]
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_.view(-1,seq_len))
        
        return final_dist, s_t, c_t, attn_dist#, p_gen

class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self,opt,init_memory=None,memory_mask=None,attention_mode = 'general',):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()


        self.attention_read = Attention(opt.rnn_size, opt.rnn_size, method=attention_mode)
        # self.memory_mask = memory_mask
        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        # self.register_buffer('mem_bias', init_memory)

        self.W = nn.Parameter(torch.FloatTensor(opt.rnn_size, opt.rnn_size))
        
        init.xavier_normal_(self.W)
        # self.WU.weight = self.WF.weight
    def set_memory(self,memory,mask):
        self.memory = memory
        self.memory_mask = mask
        _,self.N,self.M = memory.size()

    def size(self):
        return self.memory.size()

    def Read(self, q_t):
        """Read from memory (according to section 3.1)."""
        
        context,attn_dist,_ = self.attention_read(q_t.transpose(0,1),self.memory,self.memory_mask)
        return context

    def Write(self,  s_t):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.size())
        _,w,_ = self.attention_read(s_t.transpose(0,1),self.prev_mem,self.memory_mask)
        
        # print(s_t.size(),"s_t")
        F_t = torch.sigmoid(torch.matmul(s_t.transpose(0,1),self.W)).repeat(1,self.N,1)#forget
        U_t = torch.sigmoid(torch.matmul(s_t.transpose(0,1),self.W)).repeat(1,self.N,1)#add
        erase = F_t*w.transpose(1,2)
        add = U_t*w.transpose(1,2)
        self.memory = self.prev_mem * (1 - erase) + add
    