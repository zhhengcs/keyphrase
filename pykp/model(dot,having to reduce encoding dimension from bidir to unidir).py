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

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

import time

def time_usage(func):
    # argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.__name__

    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        # print(fname, "elapsed time: %f" % (end_ts - beg_ts))
        return retval

    return wrapper

class Attention(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return torch.nn.functional.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class SoftDotAttention(nn.Module):
    def __init__(self, enc_dim, trg_dim):
        super(SoftDotAttention, self).__init__()
        self.linear_in  = nn.Linear(trg_dim, trg_dim, bias=False)
        self.linear_ctx = nn.Linear(enc_dim, trg_dim)

        self.attn = nn.Linear(enc_dim + trg_dim, trg_dim)
        self.v = nn.Parameter(torch.FloatTensor(1, trg_dim))
        self.softmax = nn.Softmax()

        # input size is trg_dim * 2 as it's Dot Attention
        self.linear_out = nn.Linear(trg_dim * 2, trg_dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.method = 'concat'

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1)) # [hidden=(batch_size, dec_hidden_dim), encoder_output=(batch_size, enc_hidden_dim)] -> (batch_size, dec_hidden_dim)
            energy = torch.matmul(energy, self.v.t()) # return the energy of the k time step for all srcs in batch (batch_size, dec_hidden_dim) * (dec_hidden_dim, 1) -> (batch_size, 1)
            return energy

    def forward(self, hidden, encoder_outputs):
        '''
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim) as this is dot attention, you have to convert enc_dim to trg_dim first
        :return:
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
        batch_size      = hidden.size(0)
        src_len         = encoder_outputs.size(1)
        trg_len         = hidden.size(1)
        context_dim     = encoder_outputs.size(2)
        trg_hidden_dim  = hidden.size(2)

        # hidden (batch_size, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
        attn_energies = torch.bmm(hidden, encoder_outputs.transpose(1, 2))

        # Normalize energies to weights in range 0 to 1, (batch_size, src_len)
        # attn = torch.nn.functional.softmax(attn_energies.view(-1, encoder_outputs.size(1))) # correct attention, normalize after reshaping
        #  (batch_size, trg_len, src_len)
        attn_weights = torch.nn.functional.softmax(attn_energies.view(-1, src_len), dim = 1).view(batch_size, trg_len, src_len) # wrong attention, normalize before reshaping, but it's working

        # reweighting context, attn (batch_size, trg_len, src_len) * encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size, trg_len, src_hidden_dim)
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
        h_tilde = torch.cat((weighted_context, hidden), 1) # batch_size * (src_hidden_dim + trg_hidden_dim)
        h_tilde = self.tanh(self.linear_out(h_tilde)) # batch_size * trg_hidden_dim
        # print("--- %s seconds ---" % (time.time() - start_time))

        return h_tilde, attn

class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        attention_mode,
        batch_size,
        pad_token_src,
        pad_token_trg,
        bidirectional=True,
        nlayers_src=2,
        nlayers_trg=2,
        dropout=0.,
        must_teacher_forcing=False,
        teacher_forcing_ratio=0.,
        scheduled_sampling=False,
        scheduled_sampling_batches=2000,
    ):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()
        self.vocab_size             = vocab_size
        self.emb_dim                = emb_dim
        self.num_directions         = 2 if bidirectional else 1
        self.src_hidden_dim         = src_hidden_dim
        self.trg_hidden_dim         = trg_hidden_dim
        self.ctx_hidden_dim         = ctx_hidden_dim
        self.pad_token_src          = pad_token_src
        self.pad_token_trg          = pad_token_trg
        self.attention_mode         = attention_mode
        self.batch_size             = batch_size
        self.bidirectional          = bidirectional
        self.nlayers_src            = nlayers_src
        self.dropout                = dropout
        self.must_teacher_forcing   = must_teacher_forcing
        self.teacher_forcing_ratio  = teacher_forcing_ratio
        self.scheduled_sampling     = scheduled_sampling
        self.scheduled_sampling_batches = scheduled_sampling_batches
        self.scheduled_sampling_type= 'inverse_sigmoid' # decay curve type: linear or inverse_sigmoid
        self.current_batch          = 0

        if scheduled_sampling:
            logging.info("Applying scheduled sampling with %s decay for the first %d batches" % (self.scheduled_sampling_type, self.scheduled_sampling_batches))
        else:
            if self.must_teacher_forcing or self.teacher_forcing_ratio >= 1:
                logging.info("Training with All Teacher Forcing")
            elif self.teacher_forcing_ratio <= 0:
                logging.info("Training with All Sampling")
            else:
                logging.info("Training with Teacher Forcing with static rate=%f" % self.teacher_forcing_ratio)

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            self.pad_token_src
        )

        self.encoder = nn.LSTM(
            input_size      = emb_dim,
            hidden_size     = self.src_hidden_dim,
            num_layers      = nlayers_src,
            bidirectional   = bidirectional,
            batch_first     = True,
            dropout         = self.dropout
        )

        self.decoder = nn.LSTM(
            input_size      = emb_dim,
            hidden_size     = self.trg_hidden_dim,
            num_layers      = nlayers_trg,
            bidirectional   = False,
            batch_first     = False,
            dropout         = self.dropout
        )

        self.attention_layer = SoftDotAttention(self.src_hidden_dim * self.num_directions, trg_hidden_dim)

        self.encoder2decoder_hidden = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        self.encoder2decoder_cell = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        self.decoder2vocab = nn.Linear(trg_hidden_dim, vocab_size)

        self.init_weights()

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

        if torch.cuda.is_available():
            return h0_encoder.cuda(), c0_encoder.cuda()

        return h0_encoder, c0_encoder

    def init_decoder_state(self, enc_h, enc_c):
        # prepare the init hidden vector for decoder, (batch_size, num_layers * num_directions * enc_hidden_dim) -> (num_layers * num_directions, batch_size, dec_hidden_dim)
        decoder_init_hidden = nn.Tanh()(self.encoder2decoder_hidden(enc_h)).unsqueeze(0)
        decoder_init_cell   = nn.Tanh()(self.encoder2decoder_cell(enc_c)).unsqueeze(0)

        return decoder_init_hidden, decoder_init_cell


    def forward(self, input_src, input_src_len, input_trg, trg_mask=None, ctx_mask=None):
        '''
        To be compatible with the Copy Model, we change the output of logits to log_probs
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
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_log_probs, decoder_hiddens, attn_weights = self.decode(trg_input=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_log_probs, decoder_hiddens, attn_weights


    def encode(self, input_src, input_src_len):
        """
        Propogate input through the network.
        """
        # initial encoder state, two zero-matrix as h and c at time=0
        self.h0_encoder, self.c0_encoder = self.init_encoder_state(input_src) # (self.encoder.num_layers * self.num_directions, batch_size, self.src_hidden_dim)

        # input (batch_size, src_len), src_emb (batch_size, src_len, emb_dim)
        src_emb = self.embedding(input_src)
        # src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len, batch_first=True)

        # src_h (batch_size, seq_len, hidden_size * num_directions): outputs (h_t) of all the time steps
        # src_h_t, src_c_t (num_layers * num_directions, batch, hidden_size): hidden and cell state at last time step
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        # src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True)

        # concatenate to (batch_size, hidden_size * num_directions)
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)


    def decode(self, trg_input, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        Initial decoder state h0 (batch_size, trg_hidden_size), converted from h_t of encoder (batch_size, src_hidden_size * num_directions) through a linear layer
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
        batch_size      = trg_input.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_input.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        # maximum length to unroll
        max_length  = trg_input.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        if self.do_teacher_forcing():
            logging.info("Training batches with Teacher Forcing")
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_input = trg_input[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_input) # (batch_size, trg_len, embed_dim)
            trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, dec_hidden = self.decoder(
                trg_emb, init_hidden
            )
            # Get the h_tilde (hidden after attention) and attention weights, inputs/outputs must be batch first
            h_tildes, attn_weights, _ = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim))
            decoder_log_probs  = torch.nn.functional.log_softmax(decoder_logits, dim=-1).view(batch_size, max_length, self.vocab_size)

            decoder_outputs  = decoder_outputs.permute(1, 0, 2)

        else:
            logging.info("Training batches with All Sampling")
            # truncate the last word, as there's no further word after it for decoder to predict (batch_size, 1)
            trg_input = trg_input[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs= []
            attn_weights   = []

            dec_hidden = init_hidden
            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input) # (batch_size, trg_len, embed_dim)
                trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

                # this is trg_len first
                decoder_output, dec_hidden = self.decoder(
                    trg_emb, dec_hidden
                )

                # Get the h_tilde (hidden after attention) and attention weights, both inputs and outputs are batch first
                h_tilde, attn_weight, _ = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # (batch_size, trg_hidden_size) -> (batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))
                decoder_log_prob  = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)

                # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
                top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                top_idx = Variable(top_idx.squeeze(2))
                # top_idx and next_index are (batch_size, 1)
                trg_input = top_idx.cuda() if torch.cuda.is_available() else top_idx

                # permute to trg_len first, otherwise the cat operation would mess up things
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            decoder_outputs= torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size)
            attn_weights   = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs, hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights

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
        logging.info('coin = %f, tf_ratio = %f' % (coin, teacher_forcing_ratio))
        return coin < teacher_forcing_ratio


    def generate(self, trg_input, dec_hidden, enc_context, src_map=None, oov_list=None, max_len=1, return_attention=False):
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
        # assert isinstance(input_list, list) or isinstance(input_list, tuple)
        # assert isinstance(input_list[0], list) or isinstance(input_list[0], tuple)
        batch_size      = trg_input.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_input.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        attn_weights = []
        copy_weights = []
        log_probs = []

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        for i in range(max_len):
            # print('TRG_INPUT: %s' % str(trg_input.size()))
            # print(trg_input.data.numpy())
            trg_emb = self.embedding(trg_input)  # (batch_size, trg_len = 1, emb_dim)
            trg_emb = trg_emb.permute(1, 0, 2)  # (trg_len, batch_size, embed_dim)

            # (seq_len, batch_size, hidden_size * num_directions)
            decoder_output, dec_hidden = self.decoder(
                trg_emb, dec_hidden
            )

            # Get the h_tilde (hidden after attention) and attention weights
            h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
            # (batch_size, trg_len, trg_hidden_size) -> (batch_size, 1, vocab_size)
            decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim))

            if not hasattr(self, 'copy_model'):
                decoder_log_prob  = torch.nn.functional.log_softmax(decoder_logit, dim=-1).view(batch_size, 1, self.vocab_size)
            else:
                decoder_logit = decoder_logit.view(batch_size, 1, self.vocab_size)
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if self.copy_attention_layer:
                    _, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context)
                else:
                    copy_weight = attn_weight
                    copy_logit  = attn_logit
                copy_weights.append(copy_weight.permute(1, 0, 2)) # (1, batch_size, src_len)
                # merge the generative and copying probs (batch_size, 1, vocab_size + max_unk_word)
                decoder_log_prob   = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)

            # Prepare for the next iteration, get the top word, top_idx and next_index are (batch_size, K)
            top_1_v, top_1_idx  = decoder_log_prob.data.topk(1, dim=-1) # (batch_size, 1)
            trg_input           = Variable(top_1_idx.squeeze(2))
            # trg_input           = Variable(top_1_idx).cuda() if torch.cuda.is_available() else Variable(top_1_idx) # (batch_size, 1)

            # append to return lists
            log_probs.append(decoder_log_prob.permute(1, 0, 2)) # (1, batch_size, vocab_size)
            attn_weights.append(attn_weight.permute(1, 0, 2)) # (1, batch_size, src_len)

        # permute to trg_len first, otherwise the cat operation would mess up things
        log_probs       = torch.cat(log_probs, 0).permute(1, 0, 2) # (batch_size, max_len, K)
        attn_weights    = torch.cat(attn_weights, 0).permute(1, 0, 2) # (batch_size, max_len, src_seq_len)

        # Only return the hidden vectors of the last time step.
        #   tuple of (num_layers * num_directions, batch_size, trg_hidden_dim)=(1, batch_size, trg_hidden_dim)

        # Return final outputs, hidden states, and attention weights (for visualization)
        if return_attention:
            if not hasattr(self, 'copy_model'):
                return log_probs, dec_hidden, attn_weights
            else:
                copy_weights = torch.cat(copy_weights, 0).permute(1, 0, 2) # (batch_size, max_len, src_seq_len)
                return log_probs, dec_hidden, (attn_weights, copy_weights)
        else:
            return log_probs, dec_hidden

    def greedy_predict(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        src_h, (src_h_t, src_c_t) = self.encode(input_src)
        if torch.cuda.is_available():
            input_trg = input_trg.cuda()
        decoder_logits, hiddens, attn_weights = self.decode_old(trg_input=input_trg, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask, is_train=False)

        if torch.cuda.is_available():
            max_words_pred    = decoder_logits.data.cpu().numpy().argmax(axis=-1).flatten()
        else:
            max_words_pred    = decoder_logits.data.numpy().argmax(axis=-1).flatten()

        return max_words_pred

class Seq2SeqLSTMAttentionCopy(Seq2SeqLSTMAttention):

    def __init__(self,
            emb_dim,
            vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            ctx_hidden_dim,
            attention_mode,
            batch_size,
            pad_token_src,
            pad_token_trg,
            bidirectional=True,
            nlayers_src=2,
            nlayers_trg=2,
            dropout=0.,
            must_teacher_forcing=False,
            teacher_forcing_ratio=0.,
            scheduled_sampling=False,
            scheduled_sampling_batches=2000,
            # max_oov_number = 1000,
            unk_word = 3,
        ):
        super(Seq2SeqLSTMAttentionCopy, self).__init__(
            emb_dim,
            vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            ctx_hidden_dim,
            attention_mode,
            batch_size,
            pad_token_src,
            pad_token_trg,
            bidirectional=bidirectional,
            nlayers_src=nlayers_src,
            nlayers_trg=nlayers_trg,
            dropout=dropout,
            must_teacher_forcing=must_teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            scheduled_sampling=scheduled_sampling,
            scheduled_sampling_batches=scheduled_sampling_batches,
        )
        self.copy_model      = 'Gu'
        # self.max_oov_number   = max_oov_number
        self.unk_word  = unk_word

        if self.copy_model == 'Gu':
            self.copy_attention_layer = SoftDotAttention(self.src_hidden_dim * self.num_directions, trg_hidden_dim)
        elif self.copy_model == 'See':
            self.copy_attention_layer = None
            self.copy_gate            = nn.Linear(trg_hidden_dim, vocab_size)


    def forward(self, input_src, input_src_len, input_trg, input_src_ext, oov_lists, trg_mask=None, ctx_mask=None):
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
        src_h, (src_h_t, src_c_t) = self.encode(input_src, input_src_len)
        decoder_probs, decoder_hiddens, attn_weights, copy_attn_weights = self.decode(trg_input=input_trg, src_map=input_src_ext, oov_list=oov_lists, enc_context=src_h, enc_hidden=(src_h_t, src_c_t), trg_mask=trg_mask, ctx_mask=ctx_mask)
        return decoder_probs, decoder_hiddens, (attn_weights, copy_attn_weights)

    def merge_oov2unk(self, decoder_log_prob, max_oov_number):
        '''
        Merge the probs of oov words to the probs of <unk>, in order to generate the next word
        :param decoder_log_prob: log_probs after merging generative and copying (batch_size, trg_seq_len, vocab_size + max_oov_number)
        :return:
        '''
        batch_size, seq_len, _ = decoder_log_prob.size()
        # range(0, vocab_size)
        vocab_index     = Variable(torch.arange(start=0, end=self.vocab_size).type(torch.LongTensor))
        # range(vocab_size, vocab_size+max_oov_number)
        oov_index       = Variable(torch.arange(start=self.vocab_size, end=self.vocab_size+max_oov_number).type(torch.LongTensor))
        oov2unk_index   = Variable(torch.zeros(batch_size * seq_len, max_oov_number).type(torch.LongTensor) + self.unk_word)

        if torch.cuda.is_available():
            vocab_index   = vocab_index.cuda()
            oov_index     = oov_index.cuda()
            oov2unk_index = oov2unk_index.cuda()

        merged_log_prob = torch.index_select(decoder_log_prob, dim=2, index=vocab_index).view(batch_size * seq_len, self.vocab_size)
        oov_log_prob    = torch.index_select(decoder_log_prob, dim=2, index=oov_index).view(batch_size * seq_len, max_oov_number)

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
        :param decoder_logits: (batch_size, trg_seq_len, vocab_size)
        :param copy_logits:    (batch_size, trg_len, src_len) the pointing/copying logits of each target words
        :param src_map:        (batch_size, src_len)
        :return:
            decoder_copy_probs: return the log_probs (batch_size, trg_seq_len, vocab_size + max_oov_number)
        '''
        batch_size, max_length, _ = decoder_logits.size()
        src_len = src_map.size(1)

        # set max_oov_number to be the max number of oov
        max_oov_number   = max([len(oovs) for oovs in oov_list])

        # flatten and extend size of decoder_probs from (vocab_size) to (vocab_size+max_oov_number)
        flattened_decoder_logits = decoder_logits.view(batch_size * max_length, self.vocab_size)
        if max_oov_number > 0:
            '''
            extended_zeros           = Variable(torch.zeros(batch_size * max_length, max_oov_number))
            extended_zeros           = extended_zeros.cuda() if torch.cuda.is_available() else extended_zeros
            flattened_decoder_logits = torch.cat((flattened_decoder_logits, extended_zeros), dim=1)
            '''
            extended_logits           = Variable(torch.FloatTensor([[0.0]*len(oov)+[float('-inf')]*(max_oov_number-len(oov)) for oov in oov_list]))
            extended_logits           = extended_logits.unsqueeze(1).expand(batch_size, max_length, max_oov_number).contiguous().view(batch_size * max_length, -1)
            extended_logits           = extended_logits.cuda() if torch.cuda.is_available() else extended_logits
            flattened_decoder_logits  = torch.cat((flattened_decoder_logits, extended_logits), dim=1)

        # add probs of copied words by scatter_add_(dim, index, src), index should be in the same shape with src. decoder_probs=(batch_size * trg_len, vocab_size+max_oov_number), copy_weights=(batch_size, trg_len, src_len)
        expanded_src_map = src_map.unsqueeze(1).expand(batch_size, max_length, src_len).contiguous().view(batch_size * max_length, -1)  # (batch_size, src_len) -> (batch_size * trg_len, src_len)
        # flattened_decoder_logits.scatter_add_(dim=1, index=expanded_src_map, src=copy_logits.view(batch_size * max_length, -1))
        flattened_decoder_logits = flattened_decoder_logits.scatter_add_(1, expanded_src_map, copy_logits.view(batch_size * max_length, -1))

        # apply log softmax to normalize, ensuring it meets the properties of probability, (batch_size * trg_len, src_len)
        flattened_decoder_logits = torch.nn.functional.log_softmax(flattened_decoder_logits, dim = 1)

        # reshape to batch first before returning (batch_size, trg_len, src_len)
        decoder_log_probs = flattened_decoder_logits.view(batch_size, max_length, self.vocab_size + max_oov_number)

        return decoder_log_probs


    def decode(self, trg_input, src_map, oov_list, enc_context, enc_hidden, trg_mask, ctx_mask):
        '''
        :param
                trg_input:         (batch_size, trg_len)
                src_map  :         (batch_size, src_len), almost the same with src but oov words are replaced with temporary oov index, for copy mechanism to map the probs of pointed words to vocab words. The word index can be beyond vocab_size, e.g. 50000, 50001, 50002 etc, depends on how many oov words appear in the source text
                context vector:    (batch_size, src_len, hidden_size * num_direction) the outputs (hidden vectors) of encoder
        :returns
            decoder_probs       : (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     : (batch_size, trg_seq_len, hidden_size)
            attn_weights        : (batch_size, trg_seq_len, src_seq_len)
            copy_attn_weights   : (batch_size, trg_seq_len, src_seq_len)
        '''
        batch_size      = trg_input.size(0)
        src_len         = enc_context.size(1)
        trg_len         = trg_input.size(1)
        context_dim     = enc_context.size(2)
        trg_hidden_dim  = self.trg_hidden_dim

        # prepare the init hidden vector, (batch_size, dec_hidden_dim) -> 2 * (1, batch_size, dec_hidden_dim)
        init_hidden = self.init_decoder_state(enc_hidden[0], enc_hidden[1])

        # enc_context has to be reshaped before dot attention (batch_size, src_len, context_dim) -> (batch_size, src_len, trg_hidden_dim)
        enc_context = nn.Tanh()(self.encoder2decoder_hidden(enc_context.contiguous().view(-1, context_dim))).view(batch_size, src_len, trg_hidden_dim)

        # maximum length to unroll
        max_length  = trg_input.size(1) - 1

        # Teacher Forcing
        self.current_batch += 1
        if self.do_teacher_forcing():
            logging.info("Training batches with Teacher Forcing")
            '''
            Normal RNN procedure
            '''
            # truncate the last word, as there's no further word after it for decoder to predict
            trg_input = trg_input[:, :-1]

            # initialize target embedding and reshape the targets to be time step first
            trg_emb = self.embedding(trg_input) # (batch_size, trg_len, embed_dim)
            trg_emb  = trg_emb.permute(1, 0, 2) # (trg_len, batch_size, embed_dim)

            # both in/output of decoder LSTM is batch-second (trg_len, batch_size, trg_hidden_dim)
            decoder_outputs, hidden = self.decoder(
                trg_emb, init_hidden
            )
            # Get the h_tilde (batch_size, trg_len, trg_hidden_dim) and attention weights (batch_size, trg_len, src_len)
            h_tildes, attn_weights, attn_logits = self.attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)

            # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde), (batch_size, trg_len, trg_hidden_size) -> (batch_size * trg_len, vocab_size)
            # h_tildes=(batch_size, trg_len, trg_hidden_size) -> decoder2vocab(h_tildes.view)=(batch_size * trg_len, vocab_size) -> decoder_logits=(batch_size, trg_len, vocab_size)
            decoder_logits = self.decoder2vocab(h_tildes.view(-1, trg_hidden_dim)).view(batch_size, max_length, -1)

            '''
            Copy Mechanism
            '''
            # copy_weights and copy_logits is (batch_size, trg_len, src_len)
            if self.copy_attention_layer:
                _, copy_weights, copy_logits    = self.copy_attention_layer(decoder_outputs.permute(1, 0, 2), enc_context)
            else:
                copy_logits = attn_logits

            # merge the generative and copying probs, (batch_size, trg_len, vocab_size + max_oov_number)
            decoder_log_probs   = self.merge_copy_probs(decoder_logits, copy_logits, src_map, oov_list) # (batch_size, trg_len, vocab_size + max_oov_number)
            decoder_outputs     = decoder_outputs.permute(1, 0, 2) # (batch_size, trg_len, trg_hidden_dim)

        else:
            logging.info("Training batches with All Sampling")
            '''
            Normal RNN procedure
            '''
            # take the first word (should be BOS <s>) of each target sequence (batch_size, 1)
            trg_input = trg_input[:, 0].unsqueeze(1)
            decoder_log_probs = []
            decoder_outputs= []
            attn_weights   = []
            copy_weights   = []

            for di in range(max_length):
                # initialize target embedding and reshape the targets to be time step first
                trg_emb = self.embedding(trg_input) # (batch_size, 1, embed_dim)
                trg_emb  = trg_emb.permute(1, 0, 2) # (1, batch_size, embed_dim)

                # this is trg_len first
                decoder_output, hidden = self.decoder(
                    trg_emb, init_hidden
                )

                # Get the h_tilde (hidden after attention) and attention weights. h_tilde (batch_size,1,trg_hidden), attn_weight & attn_logit(batch_size,1,src_len)
                h_tilde, attn_weight, attn_logit = self.attention_layer(decoder_output.permute(1, 0, 2), enc_context)

                # compute the output decode_logit and read-out as probs: p_x = Softmax(W_s * h_tilde)
                # h_tilde=(batch_size, 1, trg_hidden_size) -> decoder2vocab(h_tilde.view)=(batch_size * 1, vocab_size) -> decoder_logit=(batch_size, 1, vocab_size)
                decoder_logit = self.decoder2vocab(h_tilde.view(-1, trg_hidden_dim)).view(batch_size, 1, -1)

                '''
                Copy Mechanism
                '''
                # copy_weights and copy_logits is (batch_size, trg_len, src_len)
                if self.copy_attention_layer:
                    _, copy_weight, copy_logit = self.copy_attention_layer(decoder_output.permute(1, 0, 2), enc_context)
                else:
                    copy_weight = attn_weight
                    copy_logit = attn_logit

                # merge the generative and copying probs (batch_size, 1, vocab_size + max_oov_number)
                decoder_log_prob   = self.merge_copy_probs(decoder_logit, copy_logit, src_map, oov_list)

                '''
                Find the next word
                '''
                # (deprecated, should not merge)before locating the topk, we need to move the probs of oovs to <unk>
                # oov2unk_prob = self.merge_oov2unk(decoder_log_prob, max_oov_number)
                top_v, top_idx = decoder_log_prob.data.topk(1, dim=-1)
                # replace the oov words to <unk>
                top_idx[top_idx >= self.vocab_size] = self.unk_word
                top_idx = Variable(top_idx.squeeze(2))
                # top_idx and next_index are (batch_size, 1)
                trg_input = top_idx.cuda() if torch.cuda.is_available() else top_idx

                # permute to trg_len first, otherwise the cat operation would mess up things
                decoder_log_probs.append(decoder_log_prob.permute(1, 0, 2))
                decoder_outputs.append(decoder_output)
                attn_weights.append(attn_weight.permute(1, 0, 2))
                copy_weights.append(copy_weight.permute(1, 0, 2))

            # convert output into the right shape and make batch first
            decoder_log_probs   = torch.cat(decoder_log_probs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, vocab_size + max_oov_number)
            decoder_outputs     = torch.cat(decoder_outputs, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, hidden_size)
            attn_weights        = torch.cat(attn_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)
            copy_weights        = torch.cat(copy_weights, 0).permute(1, 0, 2)  # (batch_size, trg_seq_len, src_seq_len)

        # Return final outputs (logits after log_softmax), hidden states, and attention weights (for visualization)
        return decoder_log_probs, decoder_outputs, attn_weights, copy_weights
