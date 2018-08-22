# -*- coding: utf-8 -*-
"""
Large chunk borrowed from PyTorch DataLoader
"""

import os
import json
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torch.autograd import Variable
import collections
import re
import sys
import traceback
import threading
import itertools

if sys.version_info[0] == 2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except Exception:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class KeyphraseDataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, max_batch_example=5, max_batch_pair=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
        self.dataset     = dataset
        # used for generating one2many batches
        self.num_trgs           = [len(e['trg']) for e in dataset.examples]
        self.batch_size         = max_batch_pair
        self.max_example_number = max_batch_example
        self.num_workers        = num_workers
        self.collate_fn         = collate_fn
        self.pin_memory         = pin_memory
        self.drop_last          = drop_last

        if batch_sampler is not None:
            if max_batch_pair > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        batch_sampler = One2ManyBatchSampler(sampler, self.num_trgs, 
                        max_batch_example=max_batch_example, 
                        max_batch_pair=max_batch_pair, 
                        drop_last=drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)

    def one2one_number(self):
        return sum(self.num_trgs)

class One2ManyBatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.
    Return batches of one2many pairs of which the sum of target sequences should not exceed the batch_size
    For example, if batch_size is 20 and a list of 7 examples whose number of targets are [7,5,7,6,9,7,12]
        then they are split into 4 batches: [7, 5], [7, 6], [9, 7], [12], sum of each is smaller than 20

    Original implementation is lazy loading, which cannot give the final length. Modified to load at once

    Args:
        sampler (Sampler): Base sampler.
        num_trgs (list of int): Number of target sequences for each example
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, num_trgs, max_batch_example, max_batch_pair, drop_last):
        self.sampler            = sampler
        self.num_trgs           = num_trgs
        self.max_batch_pair     = max_batch_pair
        self.max_batch_example  = max_batch_example
        self.drop_last          = drop_last

        batches = []
        batch = []
        for idx in self.sampler:
            # number of targets sequences in current batch
            number_trgs = sum([self.num_trgs[id] for id in batch])
            if len(batch) < self.max_batch_example and number_trgs + self.num_trgs[idx] < self.max_batch_pair:
                batch.append(idx)
            elif len(batch) == 0: # if the batch_size is very small, return a batch of only one data sample
                batch.append(idx)
                batches.append(batch)
                
                batch = []
            else:
                batches.append(batch)
                
                batch = []
                batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)

        self.batches         = batches
        self.final_num_batch = len(batches)
        # print(len(batches[0]),'batch len')
        # print(self.max_batch_example,'max_batch_example')
        # # print(self.num_trgs,'num_trgs')
        # print(self.max_batch_pair,'max_batch_pair')
    def __iter__(self):
        return self.batches.__iter__()

    def __len__(self):
        return self.final_num_batch

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'


def KeyphraseDataset(datapath, word2id, id2word,include_original=False):
    
    def preocess_example(e):
        keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list']
        if include_original:
            keys += ['src_str','trg_str']

        filtered_example = {}
        for k in keys:
            filtered_example[k] = e[k]
        if 'oov_list' in filtered_example:
            if type == 'one2one':
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            elif type == 'one2many':
                filtered_example['oov_number'] = [len(oov) for oov in filtered_example['oov_list']]
        filtered_example['src_len'] = len(filtered_example['src'])
        return filtered_example
    
    f = open(datapath)
    for line in f:
        sample = json.loads(line.strip())
        yield preocess_example(sample)
    
class BucketIterator(object):
    """docstring for BucketIterator"""
    def __init__(self,datapath,word2id,id2word,
            batch_size=10,sort=False,
            shuffle=None,repeat=False,
            mode='train',
            include_original=False,
            length=None):

        super(BucketIterator, self).__init__()#dataset=dataset,sort=False,shuffle=False,
                                            #batch_size=batch_size,repeat=repeat)
        self.datapath = datapath
        
        self.word2id = word2id
        self.id2word = id2word
        self.batch_size = batch_size

        # self._iterations_this_epoch = 0
        # self._random_state_this_epoch = None
        # self._restored_from_state = False
        self.include_original = include_original        
        self.pad_id = word2id[PAD_WORD]
        
        self.repeat = repeat
        self.sort = sort
        self.length=length
        self.mode = mode
    
    def __len__(self):
        return self.length

    def init_batches(self):
        self.dataset = KeyphraseDataset(self.datapath,self.word2id,self.id2word,self.include_original)
        
        if self.mode == 'train':
            self.batches = self.pool(self.dataset, self.batch_size,self.collate_fn_one2one) 
        elif self.mode =='test':
            self.batches = self.pool(self.dataset, self.batch_size,self.collate_fn_one2many) 
    
    def batch(self,data, batch_size):
    
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = len(minibatch)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            
        if minibatch:
            yield minibatch
    
    def pool(self,data, batch_size,collate_fn=None):

        for p in self.batch(data, batch_size * 100):
            p_batch =  self.batch(p, batch_size)
            
            for b in p_batch:
                yield collate_fn(b)
           
    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg = [[self.word2id[BOS_WORD]] + b['trg'] + [self.word2id[EOS_WORD]] for b in batches]

        # target_for_loss: input to criterion, if it's copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_target = [b['trg'] + [self.word2id[EOS_WORD]] for b in batches]
        trg_copy_target = [b['trg_copy'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (unk words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        
        oov_lists = [b['oov_list'] for b in batches]

        if self.sort:
            src_len_order = np.argsort([len(x) for x in src])[::-1]
            src = [src[i] for i in src_len_order]
            src_oov = [src_oov[i] for i in src_len_order]
            trg = [trg[i] for i in src_len_order]
            trg_target = [trg_target[i] for i in src_len_order]
            trg_copy_target = [trg_copy_target[i] for i in src_len_order]
            oov_lists = [oov_lists[i] for i in src_len_order]

        src, src_len, src_mask = self._pad(src)
        trg, _, _ = self._pad(trg)
        trg_target, _, _ = self._pad(trg_target)
        trg_copy_target, _, _ = self._pad(trg_copy_target)
        src_oov, _, _ = self._pad(src_oov)
        


        return src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists

    def collate_fn_one2many(self, batches):
        # source with oov words replaced by <unk>
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg = [[[self.word2id[BOS_WORD]] + t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]

        # target_for_loss: input to criterion
        trg_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]
        # target for copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_copy_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg_copy']] for b in batches]
        oov_lists = [b['oov_list'] for b in batches]

        # for training, the trg_copy_target_o2o and trg_copy_target_o2m is the final target (no way to uncover really unseen words). for evaluation, the trg_str is the final target.
        if self.include_original:
            src_str = [b['src_str'] for b in batches]
            trg_str = [b['trg_str'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_len_order = np.argsort([len(s) for s in src])[::-1]
        src = [src[i] for i in src_len_order]
        src_oov = [src_oov[i] for i in src_len_order]
        trg = [trg[i] for i in src_len_order]
        trg_target = [trg_target[i] for i in src_len_order]
        trg_copy_target = [trg_copy_target[i] for i in src_len_order]
        oov_lists = [oov_lists[i] for i in src_len_order]
        if self.include_original:
            src_str = [src_str[i] for i in src_len_order]
            trg_str = [trg_str[i] for i in src_len_order]

        # pad the one2many variables
        src_o2m, src_o2m_len, _ = self._pad(src)
        trg_o2m = trg
        src_oov_o2m, _, _ = self._pad(src_oov)
        # trg_target_o2m, _, _      = self._pad(trg_target)
        trg_copy_target_o2m = trg_copy_target
        oov_lists_o2m = oov_lists

        # unfold the one2many pairs and pad the one2one variables
        src_o2o, src_o2o_len, _ = self._pad(list(itertools.chain(*[[src[idx]] * len(t) for idx, t in enumerate(trg)])))
        src_oov_o2o, _, _ = self._pad(list(itertools.chain(*[[src_oov[idx]] * len(t) for idx, t in enumerate(trg)])))
        trg_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg])))
        trg_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_target])))
        trg_copy_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_copy_target])))
        oov_lists_o2o = list(itertools.chain(*[[oov_lists[idx]] * len(t) for idx, t in enumerate(trg)]))

        assert (len(src) == len(src_o2m) == len(src_oov_o2m) == len(trg_copy_target_o2m) == len(oov_lists_o2m))
        assert (sum([len(t) for t in trg]) == len(src_o2o) == len(src_oov_o2o) == len(trg_copy_target_o2o) == len(oov_lists_o2o))
        assert (src_o2m.size() == src_oov_o2m.size())
        assert (src_o2o.size() == src_oov_o2o.size())
        assert ([trg_o2o.size(0), trg_o2o.size(1) - 1] == list(trg_target_o2o.size()) == list(trg_copy_target_o2o.size()))

        # return two tuples, 1st for one2many and 2nd for one2one (src, src_oov, trg, trg_target, trg_copy_target, oov_lists)
        if self.include_original:
            return src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m, src_str, trg_str
        else:
            return src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m

    def _pad(self,x_raw):
        x_raw = np.asarray(x_raw)
        x_lens = [len(x_) for x_ in x_raw]
        max_length = max(x_lens)  # (deprecated) + 1 to ensure at least one padding appears in the end
        # x_lens = [x_len + 1 for x_len in x_lens]
        x = np.array([np.concatenate((x_, [self.pad_id] * (max_length - len(x_)))) for x_ in x_raw])
        x = Variable(torch.stack([torch.from_numpy(x_) for x_ in x], 0)).type('torch.LongTensor')
        x_mask = np.array([[1] * x_len + [0] * (max_length - x_len) for x_len in x_lens])
        x_mask = Variable(torch.stack([torch.from_numpy(m_) for m_ in x_mask], 0))

        assert x.size(1) == max_length

        return x, x_lens, x_mask
    
        

    def __iter__(self):
        while True:
            self.init_batches()
            for idx, minibatch in enumerate(self.batches):
                yield minibatch
            if not self.repeat:
                return

if __name__ == '__main__':
    word2id,id2word,vocab = torch.load('../data/AAAI/kp20k.vocab.pt')
    
    test_one2many_loader = BucketIterator('../data/AAAI/kp20k.test.one2many.json',word2id,id2word,
                                            batch_size=opt.beam_batch,
                                            include_original=True,
                                            mode='test',
                                            repeat=False,
                                            sort=False,
                                            shuffle=False,
                                            length=18601)


    for idx, batch in enumerate(test_one2many_loader):
        if idx > 1000:
            print(idx)
    print(done)