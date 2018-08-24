import logging
import os

import sys

import time


def init_logging(logger_name, log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    # print('Making log output file: %s' % log_file)
    # print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Device options
    parser.add_argument('-use_gpu',default=True,help='Use GPU or not.')
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=150,
                        help='Word embedding for both.')

    parser.add_argument('-position_encoding', action='store_true',
                        help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and out embeddings for decoder.')
    parser.add_argument('-share_embeddings', action='store_true',
                        help="""Share the word embeddings between encoder
                         and decoder.""")

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=150,
                        help='Size of LSTM hidden states')
    # parser.add_argument('-input_feed', type=int, default=1,
    #                     help="""Feed the context vector at each time step as
    #                     additional input (via concatenation with the word
    #                     embeddings) to the decoder.""")

    parser.add_argument('-rnn_type', type=str, default='GRU',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    parser.add_argument('-input_feeding', action="store_true",
                        help="Apply input feeding or not. Feed the updated hidden vector (after attention)"
                             "as new hidden vector to the decoder (Luong et al. 2015). "
                             "Feed the context vector at each time step  after normal attention"
                             "as additional input (via concatenation with the word"
                             "embeddings) to the decoder.")

    parser.add_argument('-bidirectional',
                        action = "store_true",default=True,
                        help="whether the encoder is bidirectional")

    # Attention options
    parser.add_argument('-attention_mode', type=str, default='concat',
                        choices=['dot', 'general', 'concat'],
                        help="""The attention type to use:
                        dot or general (Luong) or concat (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attention', action="store_true",default=True,
                        help='Train a copy model.')

    parser.add_argument('-copy_mode', type=str, default='concat',
                        choices=['dot', 'general', 'concat'],
                        help="""The attention type to use: dot or general (Luong) or concat (Bahdanau)""")

    parser.add_argument('-copy_input_feeding', action="store_true",
                        help="Feed the context vector at each time step after copy attention"
                             "as additional input (via concatenation with the word"
                             "embeddings) to the decoder.")

    parser.add_argument('-reuse_copy_attn', action="store_true",
                       help="Reuse standard attention for copy (see See et al.)")

    parser.add_argument('-copy_gate', action="store_true",
                        help="A gate controling the flow from generative model and copy model (see See et al.)")

    # parser.add_argument('-coverage_attn', action="store_true",
    #                     help='Train a coverage attention layer.')
    # parser.add_argument('-lambda_coverage', type=float, default=1,
    #                     help='Lambda value for coverage by Tu:2016:ACL.')

    # parser.add_argument('-context_gate', type=str, default=None,
    #                     choices=['source', 'target', 'both'],
    #                     help="""Type of context gate to use.
    #                     Do not select for no context gate by Tu:2017:TACL.""")

    # group.add_argument('-lambda_coverage', type=float, default=1,
    #                    help='Lambda value for coverage.')

    # Cascading model options
    parser.add_argument('-cascading_model', action="store_true",default=False,
                        help='Train a copy model.')

def preprocess_opts(parser):
    # Dictionary Options
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")
    # for copy model
    parser.add_argument('-max_unk_words', type=int, default=1000,
                        help="Maximum number of unknown words the model supports (mainly for masking in loss)")

    parser.add_argument('-words_min_frequency', type=int, default=0)

    # Length filter options
    parser.add_argument('-max_src_seq_length', type=int, default=300,
                        help="Maximum source sequence length")
    parser.add_argument('-min_src_seq_length', type=int, default=20,
                        help="Minimum source sequence length")
    parser.add_argument('-max_trg_seq_length', type=int, default=8,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-min_trg_seq_length', type=int, default=None,
                        help="Minimun target sequence length to keep.")

    # Truncation options
    parser.add_argument('-src_seq_length_trunc', type=int, default=None,
                        help="Truncate source sequence length.")
    parser.add_argument('-trg_seq_length_trunc', type=int, default=None,
                        help="Truncate target sequence length.")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")
    parser.add_argument('-lower', default=True,
                        action = 'store_true', help='lowercase data')

    # Options most relevant to summarization
    parser.add_argument('-dynamic_dict', default=True,
                        action='store_true', help="Create dynamic dictionaries (for copy)")

def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', default='data/AAAI/kp20k',
                        help="""Path prefix to the ".train.pt" and
                        ".valid.pt" file path from preprocess.py""")
    parser.add_argument('-vocab', default='data/AAAI/kp20k',
                        help="""Path prefix to the ".vocab.pt"
                        file path from preprocess.py""")

    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=200,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Dropout probability; applied in LSTM stacks.")

    # Learning options
    parser.add_argument('-train_ml', action="store_true", default=True,
                        help='Train with Maximum Likelihood or not')
    parser.add_argument('-train_rl', action="store_true", default=False,
                        help='Train with Reinforcement Learning or not')
    
    parser.add_argument('-rl_method', default=0, type=int,
                        help="""0: ori, 1: running average as baseline""")
    parser.add_argument('-rl_start_epoch', default=2, type=int,
                        help="""from which epoch rl training starts""")
    # GPU

    # Teacher Forcing and Scheduled Sampling
    parser.add_argument('-must_teacher_forcing', action="store_true",
                        help="Apply must_teacher_forcing or not")
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0,
                        help="The ratio to apply teaching forcing ratio (default 0)")
    parser.add_argument('-scheduled_sampling', action="store_true",
                        help="Apply scheduled sampling or not")
    parser.add_argument('-scheduled_sampling_batches', type=int, default=10000,
                        help="The maximum number of batches to apply scheduled sampling")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.0005,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_rl', type=float, default=0.0001,
                        help="""Starting learning rate for Reinforcement Learning.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-run_valid_every', type=int, default=200,
                        help="Run validation test at this interval (every run_valid_every batches)")
    parser.add_argument('-early_stop_tolerance', type=int, default=10,
                        help="Stop training if it doesn't improve any more for serveral rounds of validation")

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="Save checkpoint at this interval.")

    parser.add_argument('-save_model_every', type=int, default=100,
                        help="Save checkpoint at this interval.")

    parser.add_argument('-report_every', type=int, default=10,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-pred_path', type=str, default="pred/%s.%s",
                        help="Path of outputs of predictions.")
    parser.add_argument('-model_path', type=str, default="model/%s.%s",
                        help="Path of checkpoints.")

    # beam search setting
    parser.add_argument('-beam_batch', type=int, default=10,
                        help='beam_batch')

    parser.add_argument('-beam_search_batch_example', type=int, default=100,
                        help='Maximum of examples for one batch, should be disabled for training')
    parser.add_argument('-beam_search_batch_size', type=int, default=100,
                        help='Maximum batch size')
    parser.add_argument('-beam_search_batch_workers', type=int, default=6,
                        help='Number of workers for generating batches')

    parser.add_argument('-beam_size',  type=int, default=30,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=6,
                        help='Maximum sentence length.')

def predict_opts(parser):
    parser.add_argument('-must_appear_in_src', action="store_true", default="False",
                        help="""whether the predicted sequences must appear in the source text""")
    parser.add_argument('-num_oneword_seq', type=int, default=10000,
                        help="""Source sequence to decode (one line per
                        sequence)""")
    parser.add_argument('-report_score_names', type=str, nargs='+', default=['f_score@5#oneword=-1', 'f_score@10#oneword=-1'], help="""Default measure to report""")


def transformer_opts(parser):

    # transformer Options
    parser.add_argument('-d_model', type=int, default=64,
                        help='transformer hidden size every multi-head')
    parser.add_argument('-n_head', type=int, default=4,
                        help='n_head*d_model = word_vec_size')
    parser.add_argument('-d_k', type=int, default=16,
                        help='transformer hidden size every multi-head after an Linear')

    parser.add_argument('-d_v', type=int, default=16,
                        help='transformer hidden size every multi-head after an Linear')
    parser.add_argument('-d_inner_hid', type=int, default=64,
                        help='transformer hidden size in position feed forward layer')
    
    parser.add_argument('-d_layer', type=int, default=1,
                        help='the number of transformer layers')