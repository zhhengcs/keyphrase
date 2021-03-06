# -*- coding: utf-8 -*-
"""
Python File Template 
"""
from __future__ import print_function
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

def init_weights(rnn):
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

class Attention(nn.Module):
    def __init__(self, enc_dim, trg_dim, method='general'):
        super(Attention, self).__init__()
        self.method = method
        
        if self.method == 'general':
            self.W = nn.Linear(enc_dim, trg_dim,bias=False)

        elif self.method == 'concat':
            # input size is enc_dim + trg_dim as it's a concatenation of both context vectors and target hidden state            
            attn = nn.Linear(enc_dim + trg_dim, trg_dim)
            v = nn.Linear(trg_dim, 1)
            self.attn = TimeDistributedDense(mlp=attn)
            self.v = TimeDistributedDense(mlp=v)
            

        self.softmax = nn.Softmax()

        self.tanh = nn.Tanh()

    def score(self, hiddens, encoder_outputs, encoder_mask=None):
        '''
        :param hiddens: (batch, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        if self.method == 'dot':
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            if encoder_mask is not None:
                encoder_outputs =  encoder_outputs * encoder_mask.view(encoder_mask.size(0), encoder_mask.size(1), 1)
            energies = torch.bmm(hiddens, encoder_outputs.transpose(1, 2))  # (batch, trg_len, src_len)

        elif self.method == 'general':
            
            energies = self.W(encoder_outputs)  # (batch, src_len, trg_hidden_dim) ht*W*hs
            if encoder_mask is not None:
                energies =  energies * encoder_mask.view(encoder_mask.size(0), encoder_mask.size(1), 1)
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            
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

    def forward(self, hidden,encoder_outputs,encoder_value,encoder_mask=None):
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
        Q = hidden
        K = encoder_outputs
        V = encoder_value
        attn_energies = self.score(Q, K,encoder_mask)

        # Normalize energies to weights in range 0 to 1, with consideration of masks
        if encoder_mask is None:
            attn_weights = torch.nn.functional.softmax(attn_energies.view(-1, src_len), dim=1).view(batch_size, trg_len, src_len)  # (batch_size, trg_len, src_len)
        else:
            attn_energies = attn_energies * encoder_mask.view(encoder_mask.size(0), 1, encoder_mask.size(1))  # (batch, trg_len, src_len)
            attn_weights = masked_softmax(attn_energies, encoder_mask.view(encoder_mask.size(0), 1, encoder_mask.size(1)), -1)  # (batch_size, trg_len, src_len)

        # reweighting context, attn (batch_size, trg_len, src_len) * encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size, trg_len, src_hidden_dim)
        
        context = torch.bmm(attn_weights, V)

        # get h_tilde by = tanh(W_c[c_t, h_t]), both hidden and h_tilde are (batch_size, trg_hidden_dim)
        # (batch_size, trg_len=1, src_hidden_dim + trg_hidden_dim)
        # h_tilde = torch.cat((weighted_context, hidden), 2)
        # (batch_size * trg_len, src_hidden_dim + trg_hidden_dim) -> (batch_size * trg_len, trg_hidden_dim)
        # h_tilde = self.tanh(self.linear_out(h_tilde.view(-1, context_dim + trg_hidden_dim)))

        # return h_tilde (batch_size, trg_len, trg_hidden_dim), attn (batch_size, trg_len, src_len) and energies (before softmax)
        return context.view(batch_size, trg_len, trg_hidden_dim), attn_weights, attn_energies


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
        self.encoder_name = 'Attention' # BiGRU or Attention
        self.decoder_name='Memory Network' #CopyRNN or Memory Network

        self.get_mask = GetMask(self.pad_token_src)
        # self.x_context = nn.Linear(self.src_hidden_dim*2 , self.src_hidden_dim)
        
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_token_src
        )

        self.seq_encoder = Dynamic_RNN(input_size=self.emb_dim,hidden_size=self.src_hidden_dim/2,use_gpu=opt.use_gpu)
        self.matching_gru = Dynamic_RNN(self.src_hidden_dim*2,self.src_hidden_dim/2,use_gpu=opt.use_gpu)
        
        self.RNN_decoder = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.trg_hidden_dim,
            num_layers=self.nlayers_trg,
            bidirectional=False,
            batch_first=False,
            dropout=self.dropout
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.vocab_size)
        self.attention_layer = Attention(self.src_hidden_dim *1, self.trg_hidden_dim, method=self.attention_mode)
        
        self.Cross_attention_layer = Attention(self.src_hidden_dim,self.src_hidden_dim,method=self.attention_mode)
        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim*self.num_directions,
            self.trg_hidden_dim
        )

        # copy attention
        if self.copy_attention:
            if self.copy_mode == None and self.attention_mode:
                self.copy_mode = self.attention_mode
            assert self.copy_mode != None
            assert self.unk_word != None
            logging.info("Applying Copy Mechanism, type=%s" % self.copy_mode)
            # for Gu's model
            self.copy_attention_layer = Attention(self.src_hidden_dim * 1, self.trg_hidden_dim, method=self.copy_mode)
            # for See's model
            self.copy_gate = nn.Linear(self.trg_hidden_dim, self.vocab_size)
        else:
            self.copy_mode = None
            self.copy_input_feeding = False
            self.copy_attention_layer = None

        # setup for input-feeding, add a bridge to compress the additional inputs. Note that input-feeding cannot work with teacher-forcing
        self.dec_input_dim = self.emb_dim  # only input the previous word
        
        if self.input_feeding:
            self.dec_input_dim += self.trg_hidden_dim

        if self.copy_input_feeding:
            self.dec_input_dim += self.trg_hidden_dim

        if self.dec_input_dim == self.emb_dim:
            self.dec_input_bridge = None
        else:
            self.dec_input_bridge = nn.Linear(self.dec_input_dim, self.emb_dim)

        self.init_weights()

        
        if self.decoder_name == 'Memory Network':
            self.Memory_Decoder = MeMDecoder(opt)
            self.Memory_Decoder.embedding = self.embedding
            
    def init_embedding(self,embedding,requires_grad=True):
        if embedding is not None:
            if self.use_gpu:
                embedding = embedding.cuda()
            self.embedding.weight = nn.Parameter(embedding)
            self.embedding.weight.requires_grad = requires_grad
        else:
            self.embedding.weight.data.uniform_(-0.1,0.1)

    def init_weights(self):
        """Initialize weights."""
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        self.encoder2decoder.bias.data.fill_(0)
        

    def init_encoder_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)

        h_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

 
        if torch.cuda.is_available() and self.use_gpu:
            return h_encoder.cuda()#, c0_encoder.cuda()

        return h_encoder#, c0_encoder

    def init_decoder_state(self, enc_h, enc_c=None):
        # prepare the init hidden vector for decoder, (batch_size, num_layers * num_directions * enc_hidden_dim) -> (num_layers * num_directions, batch_size, dec_hidden_dim)
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder(enc_h))
        
        if self.bidirectional:
            enc_c = torch.cat([enc_c[0],enc_c[1]],-1)

        decoder_init_state = nn.Tanh()(self.encoder2decoder(enc_c)).unsqueeze(0)

        return decoder_init_hidden, decoder_init_state

    def forward(self, input_src, input_src_len, input_trg, input_src_ext, oov_lists, 
                    trg_mask=None, ctx_mask=None,query = None,query_len=None):
        
        '''
        The differences of copy model from normal seq2seq here are:
        1. The size of decoder_logits is (batch_size, trg_seq_len, vocab_size + max_oov_number).Usually vocab_size=50000 and max_oov_number=1000. And only very few of (it's very rare to have many unk words, in most cases it's because the text is not in English)
        2. Return the copy_attn_weights as well. If it's See's model, the weights are same to attn_weights as it reuse the original attention
        3. Very important: as we need to merge probs of copying and generative part, thus we have to operate with probs instead of logits. Thus here we return the probs not logits. Respectively, the loss criterion outside is NLLLoss but not CrossEntropyLoss any more.
        :param
            input_src : numericalized source text, oov words have been replaced with <unk>
            input_trg : numericalized target text, oov words have been replaced with <unk>
            input_src_ext : numericalized source text in extended vocab, oov words have been replaced with temporary oov index,
                            for copy mechanism to map the probs of pointed words to vocab words
        :returns
            decoder_logits      : (batch_size, trg_seq_len, vocab_size)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''

        if not ctx_mask:
            ctx_mask = self.get_mask(input_src)  # same size as input_src
            query_mask = self.get_mask(query)

        if self.encoder_name == 'Attention':
            encoder_outputs,encoder_hidden,query_outputs = self.attention_encode(input_src,input_src_len,ctx_mask,query,query_len,query_mask)
            
        elif self.encoder_name == 'BiGRU':
            encoder_outputs, encoder_hidden = self.encode(input_src, input_src_len)

        if self.decoder_name == 'Memory Network':
            assert self.encoder_name !='BiGRU'

            self.Memory_Decoder.ntm.set_memory(query_outputs,query_mask,encoder_outputs,ctx_mask)

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

    
    def attention_encode(self,input_src,input_src_len,src_mask,input_query,input_query_len,query_mask):
        
        src_emb = self.embedding(input_src)
        query_emb = self.embedding(input_query)
        
        src_hidden,src_state = self.seq_encoder(src_emb,input_src_len)
        query_hidden,query_state = self.seq_encoder(query_emb,input_query_len)

        context_query,_,_ = self.Cross_attention_layer(src_hidden,query_hidden,query_hidden,query_mask)
        x_with_query = torch.cat([src_hidden,context_query],-1)

        context_doc,_,_ = self.Cross_attention_layer(query_hidden,src_hidden,src_hidden,src_mask)
        query_with_x = torch.cat([query_hidden,context_doc],-1)

        query_output,query_state = self.matching_gru(query_with_x,input_query_len)
        src_output,src_state = self.matching_gru(x_with_query,input_src_len)
        
        p = 0.5
        src_output = src_hidden*p + (1-p)*src_output
        query_output = query_hidden*p + (1-p)*query_output

        src_state = torch.cat([src_state[0],src_state[1]],-1).unsqueeze(0)
        query_state = torch.cat([query_state[0],query_state[1]],-1).unsqueeze(0)

        merge_state = self.encoder2decoder(torch.cat([src_state,query_state],-1))
        
        return src_output,merge_state,query_output

    def memory_decode(self,trg_inputs,src_map,oov_list,encoder_outputs,dec_hidden,trg_mask,ctx_mask,oov_lists):
        
        s_t_1 = dec_hidden
        
        decoder_probs = []
        decoder_hiddens = []
        enc_batch_extend_vocab = src_map
        
        # maximum length to unroll, ignore the last word (must be padding)
        max_dec_len = trg_inputs.size(1) - 1
        batch_size,src_len,context_dim = encoder_outputs.size()
        _,_,trg_hidden_dim = dec_hidden.size()

        
        c_t_1,_,_ = self.Memory_Decoder.read_src(s_t_1.transpose(0,1),encoder_outputs,encoder_outputs,ctx_mask)
        
        if torch.cuda.is_available() and self.use_gpu:
            c_t_1 = c_t_1.cuda()
            
        # if self.attention_layer.method == 'dot':
        #     encoder_outputs = nn.Tanh()(self.encoder2decoder(encoder_outputs.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)
        #     encoder_outputs = encoder_outputs * ctx_mask.view(ctx_mask.size() + (1,))

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
        # print(decoder_probs.size())
        decoder_probs = torch.log(decoder_probs+1e-12)
        
        return decoder_probs,decoder_hiddens,None,None
        

    def memory_generate(self,trg_inputs,dec_hidden,encoder_outputs,ctx_mask,src_map,oov_lists,max_len=1,return_attention=False):
        
        s_t_1 = dec_hidden
        decoder_probs = []
        decoder_hiddens = []
        enc_batch_extend_vocab = src_map

        batch_size,_ = trg_inputs.size()

        # c_t_1 = Variable(torch.zeros(batch_size,1,self.ctx_hidden_dim))
        c_t_1,_,_ = self.Memory_Decoder.read_src(s_t_1.transpose(0,1),encoder_outputs,encoder_outputs,ctx_mask)
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

        src_emb = self.embedding(input_src)
        
        src_h,src_state = self.seq_encoder(src_emb,input_src_len)
        src_state = torch.cat([src_state[0],src_state[1]],-1).unsqueeze(0)


        return src_h, src_state

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
            enc_context = nn.Tanh()(self.encoder2decoder(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)
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
            
            decoder_outputs, dec_hidden = self.RNN_decoder(
                trg_emb, init_hidden
            )
            '''
            (2) Standard Attention
            '''
            # Get the h_tilde (batch_size, trg_len, trg_hidden_dim) and attention weights (batch_size, trg_len, src_len)
            h_tildes, attn_weights, attn_logits = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context,enc_context, encoder_mask=ctx_mask)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde), (batch_size, trg_len, trg_hidden_size) -> (batch_size * trg_len, vocab_size)
            # h_tildes=(batch_size, trg_len, trg_hidden_size) -> decoder2vocab(h_tildes.view)=(batch_size * trg_len, vocab_size) -> decoder_logits=(batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim)).view(batch_size, max_length, -1)

            '''
            (3) Copy Attention
            '''
            if self.copy_attention:
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)

                copy_logits = attn_logits
                copy_weights = attn_weights

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
        # flattened_decoder_probs = torch.nn.functional.softmax(flattened_decoder_logits,dim=-1)

        # reshape to batch first before returning (batch_size, trg_len, src_len)
        decoder_log_probs = flattened_decoder_logits.view(batch_size, max_length, self.vocab_size + max_oov_number)
        # decoder_probs = flattened_decoder_probs.view(batch_size,max_length,self.vocab_size+max_oov_number)
        return decoder_log_probs#,decoder_probs

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
            decoder_output, dec_hidden = self.RNN_decoder(
                dec_input, dec_hidden
            )
            
            # Get the h_tilde (hidden after attention) and attention weights
            h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context, enc_context,encoder_mask=ctx_mask)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
            decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))

            if not self.copy_attention:
                decoder_log_prob = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
            else:
                decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                
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

class Seq2SeqLSTMAttentionCascading(Seq2SeqLSTMAttention):
    def __init__(self, opt):
        super(Seq2SeqLSTMAttentionCascading, self).__init__(opt)


class MeMDecoder(nn.Module):
    def __init__(self,opt):
        super(MeMDecoder, self).__init__()
        self.read_src = Attention(opt.rnn_size, opt.rnn_size, method=opt.attention_mode)
        self.x_context = nn.Linear(opt.rnn_size + opt.word_vec_size, opt.rnn_size)
        self.gru = nn.GRU(opt.rnn_size,opt.rnn_size,num_layers=1,batch_first=True,bidirectional=False)
        init_weights(self.gru)
        self.out1 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        self.out2 = nn.Linear(opt.rnn_size,opt.vocab_size)
        self.ntm = NTMMemory(opt)

    def show_attn(self,attn,src_input):
        pass
    

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab):
        y_t_1_embd = self.embedding(y_t_1)
        
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd.unsqueeze(1)), -1))

        # first st
        
        gru_out, s_t = self.gru(x, s_t_1)
        
        for i in range(1):
            c_t = self.ntm.Read(s_t)
            _,s_t = self.gru(c_t,s_t)
            # self.ntm.Write(s_t)
        
        c_t,attn_dist,_ = self.read_src(s_t.transpose(0,1),encoder_outputs,encoder_outputs,enc_padding_mask)
        

        output = self.out1(torch.cat([s_t,c_t.transpose(0,1)],-1).squeeze(0)) # B x vocab_size
        output = self.out2(output)
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
    def __init__(self,opt,init_memory=None,memory_mask=None):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.s_context = nn.Linear(opt.rnn_size+opt.rnn_size,opt.rnn_size)
        self.read_q = Attention(opt.rnn_size, opt.rnn_size, method=opt.attention_mode)
        self.W = nn.Parameter(torch.FloatTensor(opt.rnn_size, opt.rnn_size))
        init.xavier_normal_(self.W)
        
    def set_memory(self,query_memory,query_mask,doc_memory,doc_mask=None):
        self.query_memory = query_memory
        self.query_mask = query_mask
        # self.doc_memory = doc_memory
        # self.doc_mask = doc_mask

        _,self.N,self.M = query_memory.size()

    def size(self):
        return self.query_memory.size()

    def Read(self, s_t):
        """Read from memory (according to section 3.1)."""
        context,_,_ = self.read_q(s_t.transpose(0,1),self.query_memory,self.query_memory,self.query_mask)
        return context

    def Write(self,s_t):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.query_memory
        
        _,w,_ = self.attention(s_t.transpose(0,1),self.query_memory,self.doc_memory,self.query_mask)
        F_t = torch.sigmoid(torch.matmul(s_t.transpose(0,1),self.W)).repeat(1,self.N,1)#forget
        U_t = torch.sigmoid(torch.matmul(s_t.transpose(0,1),self.W)).repeat(1,self.N,1)#add
        erase = F_t*w.transpose(1,2)
        add = U_t*w.transpose(1,2)
        self.query_memory = self.prev_mem * (1 - erase) + add
    


class Dynamic_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=True,rnn_name='GRU',use_gpu=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        """
        super(Dynamic_RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_name = rnn_name
        self.use_gpu = use_gpu
        if self.rnn_name == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif self.rnn_name == 'LSTM':
                self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: Variable
        :param x_len: numpy array
        :return:
        """
        """sort"""
        
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = np.argsort(x_sort_idx)
        x_len = x_len[x_sort_idx]

        
        x = x[x_sort_idx]
        
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        """process using RNN"""
        out_pack, ht = self.rnn(x_emb_p, None)
        
        if self.rnn_name == 'LSTM':
            ht,ct = ht
       
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

         
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]  #
        
        """unsort: out c"""
        out = out[x_unsort_idx]
        if self.rnn_name=='LSTM':
            ct = torch.transpose(ct, 0, 1)[
                x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)
            ht = (ht,ct)

        return out, ht